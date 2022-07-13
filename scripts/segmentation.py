from argparse import ArgumentParser
from functools import partial
import mimetypes
#from typing import Literal   (Supported in Python 3.8 and later)
from typing_extensions import Literal

import torch
import torchvision
from torchvision.models.segmentation import fcn_resnet50
from torchvision.transforms.functional import normalize
from torchvision.io import ImageReadMode, read_image, write_jpeg, write_png

import config  # for setting system path
from graph_networks.graph_transforms import transform_network, GraphTracker
import graph_io as gio
import utils

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def create_masks(network_output_image):
    """ creates masks for segmentation outputs

    Parameters
    ----------
    network_output_image: the output image from the segmentation network with shape: [num_classes, height, width] and of type long

    Returns
    -------
    The masks of all the segmentations ready for draw_segmentation_masks with shape: [num_classes, height, width] and of type bool
    """
    N, H, W = network_output_image.shape
    output_image = torch.argmax(network_output_image, 0)
    output_masks = torch.zeros((N, H, W), dtype=torch.bool)
    for i in range(N):
        output_masks[i] = output_image == i
    return output_masks


def segment(
        image: torch.Tensor,
        image_type: str,
        mask = None,
        mesh = None,
        device = 'cpu'
    ) -> torch.Tensor:
    """ performs segmentation using selection based convolution

    Parameters
    ----------
    - image: the input tensor image of shape [channels, height, width], type uint8 and range [0, 255]
    - itype: the type of input image either a 2d traditional image or a 360 degree image

    Returns
    -------
    - The original image with the segmentations overlayed
    """
    tensor_image = image.float() / 255
    normalized_tensor_image = normalize(tensor_image, IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0)
    
    reference_network = fcn_resnet50(pretrained=True)
    
    if image_type == "vanilla":
        reference_network = reference_network.to(device)
        reference_network.eval()
        result_image = reference_network(normalized_tensor_image.to(device))["out"]
        result_image = result_image.cpu().squeeze()
        seg_masks = create_masks(result_image)
        
    else:
        if mask is not None:
            mask = utils.loadMask(mask)

        if mesh is not None:
            mesh = loadMesh(mesh)

        if image_type == "2d":
            content,content_meta = gio.image2Graph(normalized_tensor_image,mask=mask,depth=6,device=device)
        elif image_type == "panorama":
            content,content_meta = gio.panorama2Graph(normalized_tensor_image,mask=mask,depth=6,device=device)
        elif image_type == "cubemap":
            content,content_meta = gio.sphere2Graph_cubemap(normalized_tensor_image,mask=mask,depth=6,device=device)
        elif image_type == "texture":
            content,content_meta = gio.texture2Graph(normalized_tensor_image,mesh,depth=6,device=device)
        elif image_type == "superpixel":
            content,content_meta = gio.superpixel2Graph(normalized_tensor_image,mask=mask,depth=6,device=device)
        elif image_type == "sphere":
            content,content_meta = gio.sphere2Graph(normalized_tensor_image,mask=mask,depth=6,device=device)
        elif image_type == "texture3D":
            content,content_meta = gio.texture2Graph_3D(normalized_tensor_image,mesh,depth=6,device=device)
        elif image_type == "mesh":
            content,content_meta = gio.mesh2Graph(normalized_tensor_image,mesh,depth=6,device=device)
        else:
            raise ValueError(f"image_type not known: {image_type}")

        graph_tracker = GraphTracker(content)
        graph_network = transform_network(reference_network)
        graph_network = graph_network.to(device)
        graph_network.eval()
        graph_result = graph_network(graph_tracker)["out"]
        result = graph_result.x

        # Save/show result
        if image_type == "cubemap":
            # Put back into equirectangular form
            result_image = gio.graph2Sphere_cubemap(result,content_meta)
        elif image_type == "texture" or image_type == "texture3D":
            # Remake Texture
            result_image = gio.graph2Texture(result,content_meta)
        elif image_type == "superpixel":
            # Paint back in superpixel segments
            result_image = gio.graph2Superpixel(result,content_meta)
        elif image_type == "sphere":
            # Interpolate Point Cloud
            result_image = gio.graph2Sphere(result,content_meta)
        elif image_type == "mesh":
            # Interpolate Point Cloud
            result_image = gio.graph2Mesh(result,content_meta)
        else:
            result_image = gio.graph2Image(result,content_meta) 

        seg_masks = create_masks(torch.tensor(result_image,dtype=torch.float).permute((2,0,1)))
        
    segmentation = torchvision.utils.draw_segmentation_masks(image, seg_masks)
    return segmentation


def main():
    """ the entry point to the program
    """
    read_rgb_image = partial(read_image, mode=ImageReadMode.RGB)

    parser = ArgumentParser()
    parser.add_argument(
        "image",
        type=read_rgb_image,
    )
    parser.add_argument(
        "--image_type",
        "-i",
        default="2d",
        choices=("2d","panorama","cubemap","texture","superpixel","sphere","texture3D","mesh","vanilla"),
        type=str,
    )
    parser.add_argument(
        "--out",
        "-o",
        default="output/out.png",
        type=str,
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
        "--device",
        default= 0 if torch.cuda.is_available() else "cpu",
        choices=list(range(torch.cuda.device_count())) + ["cpu"] or ["cpu"]
    )
    
    args = parser.parse_args()

    segmentation = segment(args.image, args.image_type, args.mask, args.mesh, args.device)

    exttype, _ = mimetypes.guess_type(args.out)
    if exttype == "image/jpeg":
        write_jpeg(segmentation, args.out)
    elif exttype == "image/png":
        write_png(segmentation, args.out)
    else:
        raise NotImplementedError(f"Unable to write to {args.out} of mimetype: {exttype}")


if __name__ == "__main__":
    main()
