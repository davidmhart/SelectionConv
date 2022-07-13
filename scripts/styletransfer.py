import config

import argparse
import utils
import graph_io as gio
from mesh_helpers import loadMesh
from clusters import *
from tqdm import tqdm,trange

from graph_networks.LinearStyleTransfer_vgg import encoder,decoder
from graph_networks.LinearStyleTransfer_matrix import TransformLayer

from graph_networks.LinearStyleTransfer.libs.Matrix import MulLayer
from graph_networks.LinearStyleTransfer.libs.models import encoder4, decoder4

import matplotlib.pyplot as plt


def styletransfer(contentpath,stylepath,device,out,image_type,mask,mesh):
    
    content_ref = utils.loadImage(contentpath)
    style_ref = utils.loadImage(stylepath, shape=(256,256))
    
    if mask is not None:
        mask = utils.loadMask(mask)
        
    if mesh is not None:
        mesh = loadMesh(mesh)
    
    if image_type == "2d":
        content,content_meta = gio.image2Graph(content_ref,mask=mask,depth=3,device=device)
    elif image_type == "panorama":
        content,content_meta = gio.panorama2Graph(content_ref,mask=mask,depth=3,device=device)
    elif image_type == "cubemap":
        content,content_meta = gio.sphere2Graph_cubemap(content_ref,mask=mask,depth=3,device=device)#,face_size=240)
    elif image_type == "texture":
        content,content_meta = gio.texture2Graph(content_ref,mesh,depth=3,device=device)
    elif image_type == "superpixel":
        content,content_meta = gio.superpixel2Graph(content_ref,mask=mask,depth=3,device=device)
    elif image_type == "sphere":
        content,content_meta = gio.sphere2Graph(content_ref,mask=mask,depth=3,device=device,scale=1.0)
    elif image_type == "texture3D":
        content,content_meta = gio.texture2Graph_3D(content_ref,mesh,depth=3,device=device)
    elif image_type == "mesh":
        content,content_meta = gio.mesh2Graph(content_ref,mesh,depth=3,device=device)
    else:
        raise ValueError(f"image_type not known: {image_type}")

    style,_ = gio.image2Graph(style_ref,depth=3,device=device)

    # Load original network
    enc_ref = encoder4()
    dec_ref = decoder4()
    matrix_ref = MulLayer('r41')

    enc_ref.load_state_dict(torch.load('graph_networks/LinearStyleTransfer/models/vgg_r41.pth'))
    dec_ref.load_state_dict(torch.load('graph_networks/LinearStyleTransfer/models/dec_r41.pth'))
    matrix_ref.load_state_dict(torch.load('graph_networks/LinearStyleTransfer/models/r41.pth'))

    # Copy weights to graph network
    enc = encoder()
    dec = decoder()
    matrix = TransformLayer()

    with torch.no_grad():
        enc.copy_weights(enc_ref)
        dec.copy_weights(dec_ref)
        matrix.copy_weights(matrix_ref)

    #content = content.to(device)
    #style = style.to(device)
    enc = enc.to(device)
    dec = dec.to(device)
    matrix = matrix.to(device)

    # Run graph network
    with torch.no_grad():
        cF = enc(content)
        sF = enc(style)
        feature,transmatrix = matrix(cF['r41'],sF['r41'],
                                     content.edge_indexes[3],content.selections_list[3],
                                     style.edge_indexes[3],style.selections_list[3],
                                     content.interps_list[3] if hasattr(content,'interps_list') else None)
        result = dec(feature,content)
        result = result.clamp(0,1)
        
    # Save/show result
    if image_type == "cubemap":
        # Put back into equirectangular form
        result_image = gio.graph2Sphere_cubemap(result,content_meta)
    elif image_type == "texture" or image_type == "texture3D":
        # Remake Texture
        result_image = gio.graph2Texture(result,content_meta,view3D=True)
    elif image_type == "superpixel":
        # Paint back in superpixel segments
        result_image = gio.graph2Superpixel(result,content_meta)
    elif image_type == "sphere":
        # Interpolate Point Cloud
        result_image = gio.graph2Sphere(result,content_meta)
    elif image_type == "mesh":
        # Interpolate Point Cloud
        result_image = gio.graph2Mesh(result,content_meta,view3D=True)
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
        "stylepath",
        type=str,
        default="style_ims/style0.jpg"
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
    styletransfer(**vars(args))


if __name__ == "__main__":
    main()
