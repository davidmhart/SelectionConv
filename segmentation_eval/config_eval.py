import sys
import os

sys.path.append(os.path.split(os.path.dirname(__file__))[0])

stanford_dir = "/media/david/Seagate Expansion Drive/Datasets/Stanford2D-3DS/"
#stanford_dir = "/Data6/david/Stanford2d3ds/"
pano_mask = "segmentation_eval/pano-mask.png"
weights = "segmentation_eval/originals/segmentation_ResNet.pth"
output_dir = "segmentation_eval/outputs/"