import torch
import numpy as np
from imageio import imread, imwrite
from skimage.transform import resize
import os
from torch_geometric.utils import subgraph
from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_cdt, grey_dilation
import matplotlib.pyplot as plt
import json

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def loadJSON(file):
    with open(file) as f:
        result = json.load(f)
    return result

def saveJSON(data,file):
    with open(file, 'w') as outfile:
        json.dump(data, outfile)

def loadImage(filename, asTensor = True, imagenet_mean = False, shape = None):

    image = imread(filename)

    if image.ndim == 2:
        image = np.stack((image,image,image),axis=2)

    image = image[:,:,:3]/255 # No alpha channels

    if shape is not None:
        image = resize(image, shape)

    if imagenet_mean:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean)/std

    # Reorganize for torch
    if asTensor:
        image = np.transpose(image,(2,0,1))

        image = np.expand_dims(image,axis=0)

        return torch.tensor(image,dtype=torch.float)
    else:
        return image
    
def loadMask(filename, asTensor = True, shape = None):

    image = imread(filename)
    
    if image.ndim == 3:
        image = image[:,:,0]

    if shape is not None:
        image = resize(image, shape) 

    mask = image >= 128
        
    # Reorganize for torch
    if asTensor:
        mask = np.expand_dims(mask,axis=0)

        return torch.tensor(mask,dtype=torch.bool)
    else:
        return mask
    
def toTensor(numpy_data):
    image = np.transpose(numpy_data,(2,0,1))
    image = np.expand_dims(image,axis=0)
    return torch.tensor(image,dtype=torch.float)

def toTorch(numpy_data):
    return toTensor(numpy_data)

def toNumpy(tensor, permute=True):
    image = np.squeeze(tensor.detach().clone().cpu().numpy())
    if permute:
        image = np.transpose(image,(1,2,0))
    return image

def makeCanvas(x,original):
    
    if torch.is_tensor(original):
        original = toNumpy(original)
    
    if x.ndim > 1:
        if x.shape[-1] == original.shape[-1]:
            return original

    rows,cols,_ = original.shape
    if x.ndim > 1:
        return np.zeros((rows,cols,x.shape[-1]))
    else:
        return np.zeros((rows,cols))

def reverse_selection(s):
    return [0, 5, 6, 7, 8, 1, 2, 3, 4][s]


def extrapolate_image(im, numpy = False):

    rows,cols,ch = im.shape
    if numpy:   
        result = np.zeros((rows+1,cols+1,ch))
    else:
        result = torch.zeros((rows+1,cols+1,ch))
        
    result[:rows,:cols] = im

    # Extrapolate last column
    result[:rows,cols] = 2*result[:rows,cols-1] - result[:rows,cols-2]
    # Extrapolate last row
    result[rows] = 2*result[rows-1] - result[rows-2]

    return result

def bilinear_interpolate(im, x, y, numpy = False):

    if numpy:
    
        im = extrapolate_image(im,numpy=True)

        x = np.asarray(x)
        y = np.asarray(y)

        x0 = np.floor(x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y).astype(int)
        y1 = y0 + 1

        x0 = np.clip(x0, 0, im.shape[1]-1);
        x1 = np.clip(x1, 0, im.shape[1]-1);
        y0 = np.clip(y0, 0, im.shape[0]-1);
        y1 = np.clip(y1, 0, im.shape[0]-1);

        Ia = im[ y0, x0 ]
        Ib = im[ y1, x0 ]
        Ic = im[ y0, x1 ]
        Id = im[ y1, x1 ]

        wa = (x1-x) * (y1-y)
        wb = (x1-x) * (y-y0)
        wc = (x-x0) * (y1-y)
        wd = (x-x0) * (y-y0)

        wa = np.expand_dims(wa,axis=1)
        wb = np.expand_dims(wb,axis=1)
        wc = np.expand_dims(wc,axis=1)
        wd = np.expand_dims(wd,axis=1)
        
    else:
        im = im[0].permute((1,2,0)).float()
        im = extrapolate_image(im)
        
        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1

        x0 = torch.clip(x0, 0, im.shape[1]-1);
        x1 = torch.clip(x1, 0, im.shape[1]-1);
        y0 = torch.clip(y0, 0, im.shape[0]-1);
        y1 = torch.clip(y1, 0, im.shape[0]-1);

        Ia = im[y0,x0]
        Ib = im[y1,x0]
        Ic = im[y0,x1]
        Id = im[y1,x1]

        wa = (x1-x) * (y1-y)
        wb = (x1-x) * (y-y0)
        wc = (x-x0) * (y1-y)
        wd = (x-x0) * (y-y0)

        wa = torch.unsqueeze(wa,dim=1)
        wb = torch.unsqueeze(wb,dim=1)
        wc = torch.unsqueeze(wc,dim=1)
        wd = torch.unsqueeze(wd,dim=1)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def cosineWeighting(rows,cols):
    phi_vals = np.linspace(-np.pi/2,np.pi/2,rows)
    cosines = np.cos(phi_vals)

    result = np.zeros((rows,cols))
    for i in range(rows):
        result[i, :] = cosines[i]
    
    return result

def cross(a,b):
    # Computes the cross product of two torch tensors
    out_i = a[:,1]*b[:,2] - a[:,2]*b[:,1]
    out_j = a[:,2]*b[:,0] - a[:,0]*b[:,2]
    out_k = a[:,0]*b[:,1] - a[:,1]*b[:,0]
    
    out = torch.stack((out_i,out_j,out_k),dim=1)
    
    return out

