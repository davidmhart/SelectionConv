from argparse import ArgumentParser
import os
from multiprocessing import cpu_count
import torch
import random
import config_eval as config

import torchvision
from torchvision.transforms.functional import normalize, resize
from torchvision.models.segmentation import fcn_resnet50
from graph_networks.UNet_graph import UNet as UNet_graph
from graph_networks.UNet import UNet
import graph_io as gio

from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as tvF
from stanford2d3d import Stanford2D3DDataset
from torch.utils.data import random_split
import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from trainer_graph import Trainer, EarlyStopper
from seg_metrics import get_loss_fn
from torch_geometric.utils import accuracy, mean_iou


def rgb_transform(rgb):
    if rgb.shape[0] != 3:
        raise ValueError(f"rgb's shape is not rgb: {rgb.shape}")
    rgb = rgb.float() / 255
    rgb = tvF.normalize(rgb, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return rgb

def train_transforms(rgb, seg):
    #tqdm.tqdm.write(f"{rgb.shape} {seg.shape}")
    rgb = tvF.resize(rgb, (512, 1024))
    seg = tvF.resize(seg.unsqueeze(0).unsqueeze(0), (512, 1024), InterpolationMode.NEAREST).squeeze()
    # tqdm.tqdm.write(f"{seg.shape}")
    rgb = rgb_transform(rgb)
    # print(seg.shape)
    return rgb, seg.long()

def val_transforms(rgb, seg):
    rgb = tvF.resize(rgb, (512, 1024))
    seg = tvF.resize(seg.unsqueeze(0).unsqueeze(0), (512, 1024), InterpolationMode.NEAREST).squeeze()
    # tqdm.tqdm.write(f"{seg.shape}")
    rgb = rgb_transform(rgb)
    # print(seg.shape)
    return rgb, seg.long()


def get_graph(image,mask,image_type,depth=6,device='cpu'):
    
    if image_type == "2d":
        graph,metadata = gio.image2Graph(image,mask=mask,depth=depth,device=device)
    elif image_type == "panorama":
        graph,metadata = gio.panorama2Graph(image,mask=mask,depth=depth,device=device)
    elif image_type == "cubemap":
        graph,metadata = gio.sphere2Graph_cubemap(image,mask=mask,depth=depth,device=device,face_size=192)
    elif image_type == "superpixel":
        graph,metadata = gio.superpixel2Graph(image,mask=mask,depth=depth,device=device)
    elif image_type == "sphere":
        graph,metadata = gio.sphere2Graph(image,mask=mask,depth=depth,device=device)
    else:
        raise ValueError(f"image_type not known: {image_type}")

    return graph,metadata


def finetune_segmentation(device, batch_size, max_epochs, fold, trainsplit, logdir, image_type):
    if logdir is not None:
        writer = SummaryWriter(os.path.join("segmentation_eval/runs", logdir))
    else:
        writer = None
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    # Data
    generator = torch.Generator().manual_seed(1337)
    train_fold, test_fold = Stanford2D3DDataset.get_splits(fold)
    trainset = Stanford2D3DDataset(
        config.stanford_dir,
        datatype="pano",
        areas=train_fold,
        transforms=train_transforms
    )
    num_train = int(len(trainset) * trainsplit)
    num_val = len(trainset) - num_train
    trainset, valset = random_split(trainset, (num_train, num_val), generator)

    # Network
    reference_network = UNet(Stanford2D3DDataset.num_classes)
    reference_network.load_state_dict(torch.load(config.UNet_weights))
    
    network = UNet_graph(Stanford2D3DDataset.num_classes)
    with torch.no_grad():
        network.copy_weights(reference_network) 
        
    #network = fcn_resnet50(pretrained=False, num_classes=Stanford2D3DDataset.num_classes)
    #if device != "cpu":
    #    network = torch.nn.DataParallel(network, [0,1], device)
    network = network.to(device)
    
    # Get initial graph structure
    mask_original = torchvision.io.read_image(config.pano_mask)
    mask = resize(mask_original, (512, 1024), InterpolationMode.NEAREST).type(torch.bool)
    graph,metadata = get_graph(trainset[0][0].unsqueeze(0),mask,image_type,device=device)
    
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
        num_classes = Stanford2D3DDataset.num_classes,
        early_stoppers=[
            EarlyStopper("Val/Loss", 3)
        ]
    )

    # Training Loop
    trainer.train(
        trainset,
        valset,
        graph = graph,
        mask = mask,
        image_type = image_type,
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
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--trainsplit", type=float, default=0.8)
    parser.add_argument("--logdir", type=str, default="logs_finetuned")
    parser.add_argument("--image_type", type=str, default="2d")
    args = parser.parse_args()

    finetune_segmentation(
        args.device,
        args.batch_size,
        args.max_epochs,
        args.fold,
        args.trainsplit,
        args.logdir,
        args.image_type
    )


if __name__ == "__main__":
    main()
