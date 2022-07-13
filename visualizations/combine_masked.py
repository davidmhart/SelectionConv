import utils
import os
from graph_networks import *
#from torch_geometric.data import ClusterData, ClusterLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm,trange

import torchvision
from torchvision import models,transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader

from LinearStyleTransfer_vgg import encoder,decoder
from LinearStyleTransfer_matrix import TransformLayer

from LinearStyleTransfer.libs.Matrix import MulLayer
from LinearStyleTransfer.libs.models import encoder4, decoder4

from skimage.transform import resize

import matplotlib.pyplot as plt

import os
from imageio import imread, imwrite

original_fn = "../WorkingData/eye.jpg"
prefixes = ["masked_output_","masked_reference_","post_mreference_"]
dirs = ["paper_results/masked2a/","paper_results/masked2b/","paper_results/masked2c/"]
suffixes = ["style1.jpg","style8.jpg","style6.jpg"]
mask_fns = ["../WorkingData/eye_mask1.jpg","../WorkingData/eye_mask2.jpg","../WorkingData/eye_mask3.jpg"]

save_dir = "paper_results/masked2_combo/"
save_suffix = "example3.jpg"

for prefix in prefixes:

    result = utils.loadImage(original_fn,asTensor=False)

    for i in range(len(suffixes)):
        im = utils.loadImage(dirs[i] + prefix + suffixes[i],asTensor=False)
        mask = utils.loadImage(mask_fns[i],asTensor=False) > 0.5
    
        result[np.where(mask)] = im[np.where(mask)]
        
    imwrite(save_dir + prefix + save_suffix,result)
