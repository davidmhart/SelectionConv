import os
from imageio import imread, imwrite

import numpy as np
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_image")
parser.add_argument("outdir")
args = parser.parse_args()

image = imread(args.input_image)[:,:,:3]/255


rows,cols,ch = image.shape
view = np.hstack((image[:,cols//2:],image[:,:cols//2]))


filename, ext = os.path.splitext(os.path.basename(args.input_image))
imwrite(os.path.join(args.outdir, f"{filename}-panned.{ext}"),view)
