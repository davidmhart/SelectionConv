from argparse import ArgumentParser
import os
from multiprocessing import cpu_count
import torch
import random

from torchvision.models.segmentation import fcn_resnet50
from UNet import UNet

from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as tvF
from stanford2d3d import Stanford2D3DDataset
from torch.utils.data import random_split
import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from trainer import Trainer, EarlyStopper
from seg_metrics import get_loss_fn, mean_iou
import config

def rgb_transform(rgb):
    if rgb.shape[0] != 3:
        raise ValueError(f"rgb's shape is not rgb: {rgb.shape}")
    rgb = rgb.float() / 255
    rgb = tvF.normalize(rgb, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return rgb

def train_transforms(rgb, seg):
    #tqdm.tqdm.write(f"{rgb.shape} {seg.shape}")
    rgb = tvF.resize(rgb, (256, 256))
    seg = tvF.resize(seg.unsqueeze(0).unsqueeze(0), (256, 256), InterpolationMode.NEAREST).squeeze()
    # tqdm.tqdm.write(f"{seg.shape}")
    h, w = rgb.shape[-2:]
    target_h = target_w = 224
    if random.uniform(0, 1) > 0.5:
        rgb = torch.flip(rgb, [2])
        seg = torch.flip(seg, [1])
    new_y, new_x = (random.randint(0, h-target_h-1), random.randint(0, w-target_w-1))
    rgb = rgb[:, new_y:new_y+target_h, new_x:new_x+target_w]
    seg = seg[new_y:new_y+target_h, new_x:new_x+target_w]
    rgb = rgb_transform(rgb)
    # print(seg.shape)
    return rgb, seg.squeeze(0).long()

def val_transforms(rgb, seg):
    rgb = tvF.resize(rgb, (256, 256))
    seg = tvF.resize(seg.unsqueeze(0).unsqueeze(0), (256, 256), InterpolationMode.NEAREST).squeeze()
    rgb = tvF.center_crop(rgb, (224, 224))
    seg = tvF.center_crop(seg, (224, 224))
    rgb = rgb_transform(rgb)
    # print(seg.shape)
    return rgb, seg



def pretrain_segmentation(device, batch_size, max_epochs, fold, trainsplit, logdir):
    if logdir is not None:
        writer = SummaryWriter(os.path.join("runs", logdir))
    else:
        writer = None
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    # Data
    generator = torch.Generator().manual_seed(1337)
    train_fold, test_fold = Stanford2D3DDataset.get_splits(fold)
    trainset = Stanford2D3DDataset(
        config.stanford_dir,
        areas=train_fold,
        rgb_transform=rgb_transform,
        transforms=train_transforms
    )
    num_train = int(len(trainset) * trainsplit)
    num_val = len(trainset) - num_train
    trainset, valset = random_split(trainset, (num_train, num_val), generator)

    # Network
    network = UNet(Stanford2D3DDataset.num_classes)
    #network = fcn_resnet50(pretrained=False, num_classes=Stanford2D3DDataset.num_classes)
    if device != "cpu":
        network = torch.nn.DataParallel(network, [0,1], device)
    network = network.to(device)
    optimizer = torch.optim.AdamW(network.parameters())
    criteria = torch.nn.CrossEntropyLoss()

    # setup trainer
    trainer = Trainer(
        device=device,
        network=network,
        criteria=criteria,
        optimizer=optimizer,
        metric_fns=[
            ("Loss", get_loss_fn(criteria)),
            ("MIOU", mean_iou)
        ],
        early_stoppers=[
            EarlyStopper("Val/Loss", 3)
        ]
    )

    # Training Loop
    trainer.train(
        trainset,
        valset,
        max_epochs=max_epochs,
        batch_size=batch_size,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        writer=writer,
        logdir=logdir,
    )


def main():
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--trainsplit", type=float, default=0.8)
    parser.add_argument("--logdir", type=str, default="logs")
    args = parser.parse_args()

    pretrain_segmentation(
        args.device,
        args.batch_size,
        args.max_epochs,
        args.fold,
        args.trainsplit,
        args.logdir,
    )


if __name__ == "__main__":
    main()
