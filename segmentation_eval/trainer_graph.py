import torch
import os
import json
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import cpu_count
import tqdm
import numpy as np
import graph_io as gio

def freezeBatchNorm(model):
    for name ,child in (model.named_children()):
        if name.find('BatchNorm') != -1:
            name.eval()

def get_xy(data,gt,mask,image_type,device='cpu'):
    
    if image_type == "2d":
        x,y = gio.image2Graph(data,gt=gt,mask=mask,x_only=True,device=device)
    elif image_type == "panorama":
        x,y = gio.panorama2Graph(data,gt=gt,mask=mask,x_only=True,device=device)
    elif image_type == "cubemap":
        x,y = gio.sphere2Graph_cubemap(data,gt=gt,mask=mask,x_only=True,device=device,face_size=192)
    elif image_type == "superpixel":
        x,y = gio.superpixel2Graph(data,gt=gt,mask=mask,x_only=True,device=device)
    elif image_type == "sphere":
        x,y = gio.sphere2Graph(data,gt=gt,mask=mask,x_only=True,device=device)
    else:
        raise ValueError(f"image_type not known: {image_type}")

    return x,y

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
            num_classes=None,
            early_stoppers=None,
            collate_fn=None
        ):
        self.network = network
        self.device = device or torch.device("cpu")
        self.criteria = criteria
        self.optimizer = optimizer
        self.collate_fn = collate_fn
        self.metric_fns = metric_fns if metric_fns is not None else []
        self.num_classes = num_classes
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
                pass
                #print("Model has Nan Params")

    def train_one_epoch(self, dataset, graph, mask, image_type, batch_size=1, label="Train"):
        self.network.train()
        
        #freezeBatchNorm(self.network)
        
        loader = torch.utils.data.DataLoader(
            dataset,
            num_workers = cpu_count() - 1,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
        for x_ref, y_ref in tqdm.tqdm(loader, desc="train", leave=False):
            x,y = get_xy(x_ref,gt=y_ref,mask=mask,image_type=image_type,device=self.device)
            graph.x = x
            self.optimizer.zero_grad(set_to_none=True)
            outs = self.network(graph)#["out"]
            loss = self.criteria(outs, y.squeeze().long())
            loss.backward()
            self.optimizer.step()
            self.run_metrics(outs, y.squeeze().long(), label)
            self.check_model()

    def run_metrics(self, outs, y, label):
        for name, metric_fn in self.metric_fns:
            label_name = f"{label}/{name}"
            if label_name not in self.history:
                self.history[label_name] = []
            if name == "MIOU":
                pred = torch.argmax(outs, 1)
                metric = float(metric_fn(pred, y, self.num_classes))
            else:
                metric = float(metric_fn(outs, y))
            self.history[label_name].append(metric)

    def val_one_epoch(self, dataset, graph, mask, image_type, batch_size=1, label="Val"):
        with torch.no_grad():
            self.network.eval()
            loader = torch.utils.data.DataLoader(
                dataset,
                num_workers=cpu_count() - 1,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self.collate_fn,
            )
            for x_ref, y_ref in tqdm.tqdm(loader, desc="val", leave=False):
                x,y = get_xy(x_ref,gt=y_ref,mask=mask,image_type=image_type,device=self.device) 
                graph.x = x
                outs = self.network(graph)
                self.run_metrics(outs, y.squeeze().long(), label)

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
            graph,
            mask,
            image_type,
            max_epochs=1,
            batch_size=1,
            train_transforms=None,
            val_transforms=None,
            writer=None,
            logdir=None,
        ):
        for epoch in tqdm.trange(max_epochs):
            if logdir is not None:
                model_fn = os.path.join("segmentation_eval/runs", logdir, f"checkpoint_{epoch}.pth")
            else:
                model_fn = None
            
            if os.path.exists(model_fn):
                self.load_model(model_fn)
            else:            
                trainset.dataset.transforms=train_transforms
                self.train_one_epoch(trainset, graph, mask, image_type, batch_size)
                valset.dataset.transforms=val_transforms
                self.val_one_epoch(valset, graph, mask, image_type, batch_size)
                self.write_history(epoch, writer)
                if logdir is not None:
                    self.save_model(model_fn)
                if self.should_early_stop():
                    break
                
        self.save_model(os.path.join("runs", "final_weights.pth"))
