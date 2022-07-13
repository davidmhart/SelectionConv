import config

import argparse
import utils
import graph_io as gio
from clusters import *
from tqdm import tqdm,trange

from graph_networks.Monodepth_models import Resnet18_md
from graph_networks.Monodepth_models_graph import Resnet18_graph

import matplotlib.pyplot as plt


def depth_predict(contentpath,device,out,image_type,mask=None,mesh=None,downsample=16):

    content_ref = utils.loadImage(contentpath)
    
    if mask is not None:
        mask = utils.loadMask(mask)
    
    if image_type == "2d":
        content,content_meta = gio.image2Graph(content_ref,mask=mask,depth=6,device=device)
    elif image_type == "panorama":
        content,content_meta = gio.panorama2Graph(content_ref,mask=mask,depth=6,device=device)
    elif image_type == "360":
        content,content_meta = gio.sphere2Graph_cubemap(content_ref,mask=mask,depth=6,device=device)
    elif image_type == "texture":
        mesh = loadMesh(mesh)
        content,content_meta = gio.texture2Graph(content_ref,mesh,depth=6,device=device)
    elif image_type == "superpixel":
        content,content_meta = gio.superpixel2Graph(content_ref,mask=mask,depth=6,device=device)
    else:
        raise ValueError(f"image_type not known: {image_type}")
    
    # Load original network
    reference_network = Resnet18_md(3)
    reference_network.load_state_dict(torch.load('graph_networks/pretrained_weights/monodepth_resnet18.pth'))
    

    # Copy weights to graph network
    network = Resnet18_graph(3)

    with torch.no_grad():
        network.copy_weights(reference_network)


    content = content.to(device)
    netowrk = network.to(device)

    #saveClusters("test.pt",clusters,edge_indexes,selections_list)
    #clusters, edge_indexes, selections_listloadClusters("test.pt")
    
    # Run graph network
    with torch.no_grad():
        result = network(content.x,content.clusters,content.edge_indexes,content.selections_list)[0]

    # Show mean, not standard deviation
    result = result[:,0]
    
    # Force between 0 and 1 for convience with interacting with drawing methods
    min_val = torch.min(result)
    max_val = torch.max(result)
    result = (result - min_val) / (max_val - min_val)
        
    # Save/show result
    if image_type == "360":
        # Put back into equirectangular form
        result_image = gio.graph2Sphere_cubemap(result,content_meta)
    elif image_type == "texture":
        # Remake Texture
        result_image = gio.graph2Texture(result,content_meta)
    elif image_type == "superpixel":
        # Paint back in superpixel segments
        result_image = gio.graph2Superpixel(result,content_meta)
    else:
        result_image = gio.graph2Image(result,content_meta)
        
    plt.imsave(out,result_image)
    plt.imshow(result_image);plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "contentpath",
        type=str,
    )
    parser.add_argument(
        "--device",
        default=0 if torch.cuda.is_available() else "cpu",
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
        choices=("2d","panorama","360","texture","superpixel"),
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
    parser.add_argument(
        "--downsample",
        type=int,
        default=16,
    )
    args = parser.parse_args()
    depth_predict(**vars(args))


if __name__ == "__main__":
    main()
