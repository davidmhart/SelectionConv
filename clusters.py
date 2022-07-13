import torch
from torch_scatter import scatter
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.utils import add_self_loops, add_remaining_self_loops, remove_self_loops
from torch_geometric.nn import fps, knn
from torch_sparse import coalesce
import graph_helpers as gh
import sphere_helpers as sh
import texture_helpers as th
import mesh_helpers as mh

import math
from math import pi,sqrt


from warnings import warn

def makeImageClusters(pos2D,Nx,Ny,edge_index,selections,depth=1,device='cpu',stride=2):
    clusters = []
    edge_indexes = [torch.clone(edge_index).to(device)]
    selections_list = [torch.clone(selections).to(device)]
    
    for _ in range(depth):
        Nx = Nx//stride
        Ny = Ny//stride
        cx,cy = getGrid(pos2D,Nx,Ny)
        cluster, pos2D = gridCluster(pos2D,cx,cy,Nx)
        edge_index, selections = selectionAverage(cluster, edge_index, selections)
        
        clusters.append(torch.clone(cluster).to(device))
        edge_indexes.append(torch.clone(edge_index).to(device))
        selections_list.append(torch.clone(selections).to(device))

    return clusters, edge_indexes, selections_list

def makeCubemapClusters(pos2D,Nx,Ny,edge_index,selections,depth=1,device='cpu',stride=2,mask=None):
    clusters = []
    edge_indexes = [torch.clone(edge_index).to(device)]
    selections_list = [torch.clone(selections).to(device)]
    
    for _ in range(depth):
        Nx = Nx//stride
        Ny = Ny//stride
        cx,cy = getGrid(pos2D,Nx,Ny)
        cluster, pos2D = gridCluster(pos2D,cx,cy,Nx)
        
        # Rebuild index image to feed into graph builder
        cx = scatter(cx, cluster, dim=0, reduce='max').long()
        cy = scatter(cy, cluster, dim=0, reduce='max').long()
        index_image = torch.zeros((Ny,Nx),dtype=torch.long)
        ci = torch.arange(len(cx))
        index_image[cy,cx] = ci
        
        rows = Ny//3
        cols = Nx//4
        
        if rows != cols:
            warn("Downsample cubemap has non-square faces, graph may have errors")
        
        horiz_nodes = index_image[rows:2*rows]
        top_nodes = index_image[:rows,cols:2*cols]
        bottom_nodes = index_image[2*rows:,cols:2*cols]
        
        edge_index, selections = sh.buildCubemapEdges(horiz_nodes,top_nodes,bottom_nodes)
        
        if mask is not None:
            temp_mask = scatter(mask.float(),cluster,dim=0,reduce='max').bool()
            edge_index,selections = gh.maskGraph(temp_mask,edge_index,selections)
            cluster = cluster[torch.where(mask)]
            cluster, _ = consecutive_cluster(cluster)
            mask = temp_mask
        
        clusters.append(torch.clone(cluster).to(device))
        edge_indexes.append(torch.clone(edge_index).to(device))
        selections_list.append(torch.clone(selections).to(device))

    return clusters, edge_indexes, selections_list

    
def getGrid(pos,Nx,Ny,xrange=None,yrange=None):
    xmin = torch.min(pos[:,0]) if xrange is None else xrange[0]
    ymin = torch.min(pos[:,1]) if yrange is None else yrange[0]
    xmax = torch.max(pos[:,0]) if xrange is None else xrange[1]
    ymax = torch.max(pos[:,1]) if yrange is None else yrange[1]

    cx = torch.clamp(torch.floor((pos[:,0] - xmin)/(xmax-xmin) * Nx),0,Nx-1)
    cy = torch.clamp(torch.floor((pos[:,1] - ymin)/(ymax-ymin) * Ny),0,Ny-1)
    return cx, cy
    
def gridCluster(pos,cx,cy,xmax):
    cluster = cx + cy*xmax
    cluster = cluster.type(torch.long) # Cast appropriately
    cluster, _ = consecutive_cluster(cluster)
    pos = scatter(pos, cluster, dim=0, reduce='mean')
    
    return cluster, pos

def selectionAverage(cluster, edge_index, selections):
    num_nodes = cluster.size(0)
    edge_index = cluster[edge_index.contiguous().view(1, -1)].view(2, -1)
    edge_index, selections = remove_self_loops(edge_index, selections)
    if edge_index.numel() > 0:

        # To avoid means over discontinuities, do mean for two selections at at a time
        final_edge_index, selections_check = coalesce(edge_index, selections.type(torch.float), num_nodes, num_nodes, op="mean")
        selections_check = torch.round(selections_check).type(torch.long)

        final_selections = torch.zeros_like(selections_check).to(selections.device)

        final_selections[torch.where(selections_check==4)] = 4
        final_selections[torch.where(selections_check==5)] = 5

        #Rotate selection kernel
        selections += 2
        selections = selections % 9 + torch.div(selections, 9, rounding_mode="floor")

        _, selections_check = coalesce(edge_index, selections.type(torch.float), num_nodes, num_nodes, op="mean")
        selections_check = torch.round(selections_check).type(torch.long)
        final_selections[torch.where(selections_check==4)] = 2
        final_selections[torch.where(selections_check==5)] = 3

        #Rotate selection kernel
        selections += 2
        selections = selections % 9 + torch.div(selections, 9, rounding_mode="floor")

        _, selections_check = coalesce(edge_index, selections.type(torch.float), num_nodes, num_nodes, op="mean")
        selections_check = torch.round(selections_check).type(torch.long)
        final_selections[torch.where(selections_check==4)] = 8
        final_selections[torch.where(selections_check==5)] = 1

        #Rotate selection kernel
        selections += 2
        selections = selections % 9 + torch.div(selections, 9, rounding_mode="floor")

        _, selections_check = coalesce(edge_index, selections.type(torch.float), num_nodes, num_nodes, op="mean")
        selections_check = torch.round(selections_check).type(torch.long)
        final_selections[torch.where(selections_check==4)] = 6
        final_selections[torch.where(selections_check==5)] = 7

        #print(torch.min(final_selections), torch.max(final_selections))
        #print(torch.mean(final_selections.type(torch.float)))

        edge_index, selections = add_remaining_self_loops(final_edge_index,final_selections,fill_value=torch.tensor(0,dtype=torch.long))

    else:
        edge_index, selections = add_remaining_self_loops(edge_index,selections,fill_value=torch.tensor(0,dtype=torch.long))
        print("Warning: Edge Pool found no edges")

    return edge_index, selections
