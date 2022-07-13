import config

import argparse
import utils
import graph_io as gio
from mesh_helpers import loadMesh
from clusters import *
from tqdm import tqdm,trange

import numpy as np

import matplotlib.pyplot as plt

def compare(result_ref,result,flip=True):
    
    result_ref = utils.toNumpy(result_ref)
    result_image = result.cpu().numpy().reshape(result_ref.shape)
    if flip:
        result_image = result_image[::-1]
    plt.imshow(np.hstack((result_ref,result_image))[:,:,:3]); plt.show()
    plt.imshow(np.sum(np.abs(result_ref - result_image),axis=2)); plt.colorbar(); plt.show()

def test_net(impath,device,out,image_type,mask,mesh):
    
    im = utils.loadImage(impath)
    
    depth = 6
    
    if mask is not None:
        mask = utils.loadMask(mask)
    
    if mesh is not None:
        mesh = loadMesh(mesh)
    
    if image_type == "2d":
        graph,metadata = gio.image2Graph(im,mask=mask,depth=depth,device=device)
    elif image_type == "panorama":
        graph,metadata = gio.panorama2Graph(im,mask=mask,depth=depth,device=device)
    elif image_type == "cubemap":
        graph,metadata = gio.sphere2Graph_cubemap(im,mask=mask,depth=depth,device=device)
    elif image_type == "texture":
        graph,metadata = gio.texture2Graph(im,mesh,depth=depth,device=device)
    elif image_type == "superpixel":
        graph,metadata = gio.superpixel2Graph(im,mask=mask,depth=depth,device=device)
    elif image_type == "sphere":
        graph,metadata = gio.sphere2Graph(im,mask=mask,depth=depth,device=device)
    elif image_type == "texture3D":
        graph,metadata = gio.texture2Graph_3D(im,mesh,depth=depth,device=device)
    elif image_type == "mesh":
        graph,metadata = gio.mesh2Graph(im,mesh,depth=depth,device=device)
    else:
        raise ValueError(f"image_type not known: {image_type}")

    from graph_networks.UNet import UNet
    from graph_networks.UNet_graph import UNet as UNet_graph
    reference_network = UNet(14)
    reference_network.load_state_dict(torch.load("segmentation_eval/weights/UNet_Weights.pth"))

    network = UNet_graph(14)
    with torch.no_grad():
        network.copy_weights(reference_network) 
    
    network = network.to(device)
    reference_network = reference_network.to(device)
    
    network.eval()
    reference_network.eval()
    
    with torch.no_grad():
        #result_ref = reference_network(im.to(device))
        #result = network(graph)
        
        result_ref,intermediate_ref = reference_network(im.to(device))
        result,intermediate = network(graph)
    
    compare(intermediate_ref[0],intermediate[0],flip=False)
    compare(intermediate_ref[1],intermediate[1])
    compare(intermediate_ref[2],intermediate[2])
    compare(intermediate_ref[3],intermediate[3])
    compare(intermediate_ref[4],intermediate[4])
    compare(intermediate_ref[5],intermediate[5])
    compare(intermediate_ref[6],intermediate[6])
    compare(intermediate_ref[7],intermediate[7])
    compare(intermediate_ref[8],intermediate[8])
    compare(intermediate_ref[9],intermediate[9])
    compare(intermediate_ref[10],intermediate[10],flip=False)

    compare(result_ref,result,flip=False)

    plt.imshow(np.sum(np.abs(utils.toNumpy(result_ref) - gio.graph2Image(result,metadata)),axis=2)); plt.colorbar(); plt.show()
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "impath",
        type=str,
    )
    parser.add_argument(
        "--device",
        default= 0 if torch.cuda.is_available() else "cpu",
        choices=list(range(torch.cuda.device_count())) + ["cpu"] or ["cpu"]
    )
    parser.add_argument(
        "--out",
        "-o",
        type=str,
        default="output/output.jpg"
    )
    parser.add_argument(
        "--image_type",
        choices=("2d","panorama","cubemap","texture","superpixel","sphere","texture3D","mesh"),
        default="2d",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    test_net(**vars(args))


if __name__ == "__main__":
    main()
