import torch
from torch_scatter import scatter

def avgPoolKernel(x,edge_index,selections,cluster,kernel_size=2,even_dirs=[0,1,7,8]):

    is_even = kernel_size % 2 == 0
    full_passes = kernel_size//2 - int(is_even)

    # Assumes the lowest number node index is the topleft most position in the cluster
    indices = torch.arange(len(x)).to(x.device)

    # Find the minimum node index in each cluster and select those x values
    picks = scatter(indices, cluster, dim=0, reduce='min')

    # Send max pool messages the appropriate number of times
    for _ in range(full_passes):
        message = x[edge_index[1]]
        x = scatter(message,edge_index[0],dim=0,reduce='mean') # Aggregate

    # Even kernel_sizes are not symetric and are lopsided towards the bottom right corner
    # Repeat the process one more time going to the bottom right
    if is_even:

        # Prefilter edge_index
        keep = torch.zeros_like(selections,dtype=torch.bool).to(x.device)
        for i in even_dirs:
            keep[torch.where(selections == i)] = True
        even_edge_index = edge_index[:,torch.where(keep)[0]]

        message = x[even_edge_index[1]]
        x = scatter(message,even_edge_index[0],dim=0,reduce='mean') # Aggregate

    # Take the previously selected nodes
    x = x[picks]

    return x

def maxPoolKernel(x,edge_index,selections,cluster,kernel_size=2,even_dirs=[0,1,7,8]):

    is_even = kernel_size % 2 == 0
    full_passes = kernel_size//2 - int(is_even)

    # Assumes the lowest number node index is the topleft most position in the cluster
    indices = torch.arange(len(x)).to(x.device)

    # Find the minimum node index in each cluster and select those x values
    picks = scatter(indices, cluster, dim=0, reduce='min')

    # Send max pool messages the appropriate number of times
    for _ in range(full_passes):
        message = x[edge_index[1]]
        x = scatter(message,edge_index[0],dim=0,reduce='max') # Aggregate

    # Even kernel_sizes are not symetric and are lopsided towards the bottom right corner
    # Repeat the process one more time going to the bottom right
    if is_even:

        # Prefilter edge_index
        keep = torch.zeros_like(selections,dtype=torch.bool).to(x.device)
        for i in even_dirs:
            keep[torch.where(selections == i)] = True
        even_edge_index = edge_index[:,torch.where(keep)[0]]

        message = x[even_edge_index[1]]
        x = scatter(message,even_edge_index[0],dim=0,reduce='max') # Aggregate

    # Take the previously selected nodes
    x = x[picks]

    return x


def stridePoolCluster(x,cluster):

    # Assumes the lowest number node index is the topleft most position in the cluster
    indices = torch.arange(len(x)).to(x.device)

    # Find the minimum node index in each cluster and select those x values
    picks = scatter(indices, cluster, dim=0, reduce='min')
    x = x[picks]

    return x


def maxPoolCluster(x,cluster):

    x = scatter(x, cluster, dim=0, reduce='max')
    return x

def avgPoolCluster(x,cluster,edge_index=None, edge_weight=None):

    x = scatter(x, cluster, dim=0, reduce='mean')
    return x
    
    
def unpoolInterpolated(x,cluster,up_edge_index,up_interps=None):

    if up_interps is None:
        return unpoolEdgeAverage(x,cluster,up_edge_index)
    
    # Determine node averages based on based on interps
    target_clusters = cluster[up_edge_index[1]]
    
    node_vals = x[target_clusters]*up_interps.unsqueeze(1)
    x = scatter(node_vals,up_edge_index[0],dim=0,reduce='add')
    norm = scatter(up_interps,up_edge_index[0],dim=0)
    x/=norm.unsqueeze(1)

    return x

def unpoolBilinear(x,cluster,up_edge_index,up_selections,selection_dirs=[0,1,7,8]):

    # Remove edges that won't be used for the bilinear interpolation calculation
    keep = torch.zeros_like(up_selections,dtype=torch.bool).to(x.device)
    for i in selection_dirs:
        keep[torch.where(up_selections == i)] = True

    ref_edge_index = up_edge_index[:,torch.where(keep)[0]]

    cluster_index = torch.vstack((ref_edge_index[0],cluster[ref_edge_index[1]]))
    cluster_index = torch.unique(cluster_index,dim=1)
    x = scatter(x[cluster_index[1]],cluster_index[0],dim=0,reduce='mean')

    return x

def unpoolEdgeAverage(x,cluster,up_edge_index,weighted=True):
    # Interpolates based on the number of connections to each previous cluster. Works best with dense data.
    # If weighted = False, clusters are weighted equally regardless of the number of connections

    if weighted:
        target_clusters = cluster[up_edge_index[1]]
        x = scatter(x[target_clusters],up_edge_index[0],dim=0,reduce='mean')

    else:
        cluster_index = torch.vstack((up_edge_index[0],cluster[up_edge_index[1]]))
        cluster_index = torch.unique(cluster_index,dim=1)
        x = scatter(x[cluster_index[1]],cluster_index[0],dim=0,reduce='mean')

    return x

def unpoolCluster(x,cluster):
    
    return x[cluster]
