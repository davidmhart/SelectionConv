import sys
import os

sys.path.append(os.path.split(os.path.dirname(__file__))[0])

stanford_dir = "/media/david/Seagate Expansion Drive/Datasets/Stanford2D-3DS/"
#stanford_dir = "/Data6/david/Stanford2d3ds/"
pano_mask = "segmentation_eval/pano-mask.png"
weights = "segmentation_eval/originals/checkpoint_49.pth"
output_dir = "segmentation_eval/outputs/"

UNet_weights = "graph_networks/pretrained_weights/UNet_2d3ds.pth"
UNet_finetuned_2d_1epoch = "graph_networks/pretrained_weights/2d_1epoch.pth"
UNet_finetuned_2d_5epoch = "graph_networks/pretrained_weights/2d_5epoch.pth"
UNet_finetuned_cubemap_1epoch = "graph_networks/pretrained_weights/cubemap_1epoch.pth"
UNet_finetuned_cubemap_5epoch = "graph_networks/pretrained_weights/cubemap_5epoch.pth"
UNet_finetuned_sphere_1epoch = "graph_networks/pretrained_weights/sphere_1epoch.pth"
UNet_finetuned_sphere_5epoch = "graph_networks/pretrained_weights/sphere_5epoch.pth"
UNet_finetuned_bary_1epoch = "graph_networks/pretrained_weights/bary_1epoch.pth"