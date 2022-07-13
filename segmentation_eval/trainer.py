import torch
import os
import json
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import cpu_count
import tqdm
import numpy as np


class EarlyStopper:
    def __init__(self, monitor, patience=0, min_delta=0, check_min=True):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.check_min = check_min
        self.num_worse = 0
        self.last_monitored = np.inf if check_min else -np.inf

    def __call__(self, data):
        last_monitored = np.mean(data)
        if self.check_min and last_monitored >= self.last_monitored + self.min_delta:
            self.num_worse += 1
        elif not self.check_min and last_monitored <= last_monitored - self.min_delta:
            self.num_worse += 1

        self.last_monitored = last_monitored
        return self.num_worse > self.patience



class Trainer:
    def __init__(
            self,
            network,
            device=None,
            criteria=None,
            optimizer=None,
            metric_fns=None,
            early_stoppers=None,
            collate_fn=None
        ):
        self.network = network
        self.device = device or torch.device("cpu")
        self.criteria = criteria
        self.optimizer = optimizer
        self.collate_fn = collate_fn
        self.metric_fns = metric_fns if metric_fns is not None else []
        self.early_stoppers = early_stoppers if early_stoppers is not None else []
        self.history = {}
        self.reset_history()

    def reset_history(self):
        for name, _ in self.metric_fns:
            self.history[f"Train/{name}"] = []
            self.history[f"Val/{name}"] = []

    def save_model(self, path):
        if isinstance(self.network, torch.nn.DataParallel):
            torch.save(self.network.module.state_dict(), path)
        else:
            torch.save(self.network.state_dict(), path)

    def load_model(self, path):
        self.network.load_state_dict(torch.load(path, self.device))

    def check_model(self):
        for param in self.network.parameters():
            if torch.any(torch.isnan(param)):
                print("HERE")

    def train_one_epoch(self, dataset, batch_size=1, label="Train"):
        self.network.train()
        loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cpu_count(),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
        for x, y in tqdm.tqdm(loader, desc="train", leave=False):            
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            outs = self.network(x)#["out"]
            print(outs.shape,y.shape)
            loss = self.criteria(outs, y)
            loss.backward()
            self.optimizer.step()
            self.run_metrics(outs, y, label)
            self.check_model()

    def run_metrics(self, outs, y, label):
        for name, metric_fn in self.metric_fns:
            label_name = f"{label}/{name}"
            if label_name not in self.history:
                self.history[label_name] = []
            metric = float(metric_fn(outs, y))
            self.history[label_name].append(metric)

    def val_one_epoch(self, dataset, batch_size=1, label="Val"):
        with torch.no_grad():
            self.network.eval()
            loader = torch.utils.data.DataLoader(
                dataset,
                num_workers=cpu_count(),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self.collate_fn,
            )
            for x, y in tqdm.tqdm(loader, desc="val", leave=False):
                x, y = x.to(self.device), y.to(self.device)
                outs = self.network(x)
                self.run_metrics(outs, y, label)

    def should_early_stop(self):
        if not self.early_stoppers:
            return False
        for early_stopper in self.early_stoppers:
            key = early_stopper.monitor
            if key in self.history:
                if early_stopper(self.history[key]):
                    return True
        return False

    def write_history(self, epoch, writer: SummaryWriter, flush=True):
        if writer is not None:
            for tag, values in self.history.items():
                writer.add_scalar(tag, np.mean(values), epoch)
        if flush:
            self.reset_history()

    def train(
            self,
            trainset,
            valset,
            max_epochs=1,
            batch_size=1,
            train_transforms=None,
            val_transforms=None,
            writer=None,
            logdir=None,
        ):
        for epoch in tqdm.trange(max_epochs):
            if logdir is not None:
                model_fn = os.path.join("runs", logdir, f"checkpoint_{epoch}.pth")
            else:
                model_fn = None
            
            if os.path.exists(model_fn):
                self.load_model(model_fn)
            else:            
                trainset.dataset.transforms=train_transforms
                self.train_one_epoch(trainset, batch_size)
                valset.dataset.transforms=val_transforms
                self.val_one_epoch(valset, batch_size)
                self.write_history(epoch, writer)
                if logdir is not None:
                    self.save_model(model_fn)
                if self.should_early_stop():
                    break
                
        self.save_model(os.path.join("runs", "final_weights.pth"))
