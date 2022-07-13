import torch
from torch_geometric.nn import radius_graph, knn_graph
from torch_geometric.utils import subgraph
import utils
from math import sqrt

def getImPos(rows,cols,start_row=0,start_col=0):
    row_space = torch.arange(start_row,rows+start_row)
    col_space = torch.arange(start_col,cols+start_col)
    col_image,row_image = torch.meshgrid(col_space,row_space,indexing='xy')
    im_pos = torch.reshape(torch.stack((row_image,col_image),dim=-1),(rows*cols,2))
    return im_pos

def convertImPos(im_pos,flip_y=True):
    
    # Cast to float for clustering based methods
    pos2D = im_pos.float()
    
    # Switch rows,cols to x,y
    pos2D[:,[1,0]] = pos2D[:,[0,1]]
    
    if flip_y:
        
        # Flip to y-axis to match mathematical definition and edges2Selections settings
        pos2D[:,1] = torch.amax(pos2D[:,1]) - pos2D[:,1]
    
    return pos2D


def grid2Edges(locs):
    # Assume locs are already spaced at a distance of 1 structure
    edge_index = radius_graph(locs,1.44,loop=True)
    return edge_index
    
def radius2Edges(locs,r=1.0):
    edge_index = radius_graph(locs,r,loop=True)
    return edge_index
    
def knn2Edges(locs,knn=9):
    edge_index = knn_graph(locs,knn,loop=True)
    return edge_index

def surface2Edges(pos3D,normals,up_vector=None,k_neighbors=9):
    
    if up_vector is None:
        up_vector = torch.tensor([[0.0,1.0,0.0]]).to(pos3D.device)
    
    # K Nearest Neighbors graph
    edge_index = knn_graph(pos3D,k_neighbors,loop=True)

    # Cull neighbors based on normals (dot them together)
    culling = torch.sum(torch.multiply(normals[edge_index[1]],normals[edge_index[0]]),dim=1)
    edge_index = edge_index[:,torch.where(culling>0)[0]]

    # For each node, rotate based on Grahm-Schmidt Orthognalization
    norms = normals[edge_index[0]]
    
    z_dir = norms
    z_dir = z_dir/torch.linalg.norm(z_dir,dim=1,keepdims=True) # Make sure it is a unit vector
    #x_dir = torch.cross(up_vector,norms,dim=1)
    x_dir = utils.cross(up_vector,norms) # torch.cross doesn't broadcast properly in some versions of torch
    x_dir = x_dir/torch.linalg.norm(x_dir,dim=1,keepdims=True)
    #y_dir = torch.cross(norms,x_dir,dim=1)
    y_dir = utils.cross(norms,x_dir)
    y_dir = y_dir/torch.linalg.norm(y_dir,dim=1,keepdims=True)

    directions = (pos3D[edge_index[1]] - pos3D[edge_index[0]])
    
    # Perform rotation by multiplying out rotation matrix
    temp = torch.clone(directions) # Buffer
    directions[:,0] = temp[:,0] * x_dir[:,0] + temp[:,1] * x_dir[:,1] + temp[:,2] * x_dir[:,2]
    directions[:,1] = temp[:,0] * y_dir[:,0] + temp[:,1] * y_dir[:,1] + temp[:,2] * y_dir[:,2]
    #directions[:,2] = temp[:,0] * z_dir[:,0] + temp[:,1] * z_dir[:,1] + temp[:,2] * z_dir[:,2]
    
    # Drop z coordinate
    directions = directions[:,:2]
    
    return edge_index, directions
      
def edges2Selections(edge_index,directions,interpolated=True,bary_d=None,y_down=False):
    
    # Current Ordering
    # 4  3  2
    # 5  0  1
    # 6  7  8
    if y_down:
        vectorList = torch.tensor([[1,0],[sqrt(2)/2,-sqrt(2)/2],[0,-1],[-sqrt(2)/2,-sqrt(2)/2],[-1,0],[-sqrt(2)/2,sqrt(2)/2],[0,1],[sqrt(2)/2,sqrt(2)/2]],dtype=torch.float).transpose(1,0)
    else:
        vectorList = torch.tensor([[1,0],[sqrt(2)/2,sqrt(2)/2],[0,1],[-sqrt(2)/2,sqrt(2)/2],[-1,0],[-sqrt(2)/2,-sqrt(2)/2],[0,-1],[sqrt(2)/2,-sqrt(2)/2]],dtype=torch.float).transpose(1,0)
    
    if interpolated:    
        
        if bary_d is None:
            edge_index,selections,interps = interpolateSelections(edge_index,directions,vectorList)
        else:
            edge_index,selections,interps = interpolateSelections_barycentric(edge_index,directions,bary_d,vectorList)
        interps = normalizeEdges(edge_index,selections,interps)
        return edge_index,selections,interps
    
    else:
        selections = torch.argmax(torch.matmul(directions,vectorList),dim=1) + 1
        selections[torch.where(torch.sum(torch.abs(directions),axis=1) == 0)] = 0 # Same cell selection
        return selections

    
def makeEdges(prev_sources,prev_targets,prev_selections,sources,targets,selection,reverse=True):
    
    sources = sources.flatten()
    targets = targets.flatten()
    
    prev_sources += sources.tolist()
    prev_targets += targets.tolist()
    prev_selections += len(sources)*[selection]
    
    if reverse:
        prev_sources += targets
        prev_targets += sources
        prev_selections += len(sources)*[utils.reverse_selection(selection)]
        
    return prev_sources,prev_targets,prev_selections
        
def maskNodes(mask,x):
    node_mask = torch.where(mask)
    x = x[node_mask]
    return x

def maskPoints(mask,x,y):
    
    mask = torch.squeeze(mask)
    
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1
    
    x0 = torch.clip(x0, 0, mask.shape[1]-1);
    x1 = torch.clip(x1, 0, mask.shape[1]-1);
    y0 = torch.clip(y0, 0, mask.shape[0]-1);
    y1 = torch.clip(y1, 0, mask.shape[0]-1);

    Ma = mask[ y0, x0 ]
    Mb = mask[ y1, x0 ]
    Mc = mask[ y0, x1 ]
    Md = mask[ y1, x1 ]
    
    node_mask = torch.where(torch.logical_and(torch.logical_and(torch.logical_and(Ma,Mb),Mc),Md))[0]
    
    return node_mask
    

def maskGraph(mask,edge_index,selections,interps=None):
    
    edge_index,_,edge_mask = subgraph(mask,edge_index,relabel_nodes=True,return_edge_mask=True)
    selections = selections[edge_mask]
    
    if interps:
        interps = interps[edge_mask]
        return edge_index, selections, interps
    else:
        return edge_index, selections
    
def interpolateSelections(edge_index,directions,vectorList=None):

    if vectorList is None:
        # Current Ordering
        # 4  3  2
        # 5  0  1
        # 6  7  8
        vectorList = torch.tensor([[1,0],[sqrt(2)/2,sqrt(2)/2],[0,1],[-sqrt(2)/2,sqrt(2)/2],[-1,0],[-sqrt(2)/2,-sqrt(2)/2],[0,-1],[sqrt(2)/2,-sqrt(2)/2]],dtype=torch.float).transpose(1,0)

    # Normalize directions for simplicity of calculations
    dir_norm = torch.linalg.norm(directions,dim=1,keepdims=True)
    directions = directions/dir_norm
    #locs = torch.where(dir_norm > 1)[0]
    #directions[locs] = directions[locs]/dir_norm[locs]
    
    values = torch.matmul(directions,vectorList)
    best = torch.unsqueeze(torch.argmax(values,dim=1),1)
    
    best_val = torch.take_along_dim(values,best,dim=1)
    
    # Look at both neighbors to see who is closer
    lower_val = torch.take_along_dim(values,(best-1) % 8,dim=1)
    upper_val = torch.take_along_dim(values,(best+1) % 8,dim=1)
    
    comp_vals = torch.cat((lower_val,upper_val),dim=1)
    
    second_best_vals = torch.amax(comp_vals,dim=1)
    second_best = torch.argmax(comp_vals,dim=1)
    
    # Find the interpolation value (in terms of angles)
    best_val = torch.minimum(best_val[:,0],torch.tensor(1,device=directions.device)) # Prep for arccos function
    angle_best = torch.arccos(best_val)
    angle_second_best = torch.arccos(second_best_vals)
    
    angle_vals = angle_best/(angle_second_best + angle_best)
    
    # Use negative values for clockwise selections
    clockwise = torch.where(second_best == 0)[0]
    angle_vals[clockwise] = -angle_vals[clockwise] 
    
    # Handle computation problems at the poles
    angle_vals = torch.nan_to_num(angle_vals)
    
    # Make Selections
    selections = best[:,0] + 1
    
    # Same cell selection
    same_locs = torch.where(edge_index[0] == edge_index[1])
    selections[same_locs] = 0
    angle_vals[same_locs] = 0
    
    # Make starting interp_values
    interps = torch.ones_like(angle_vals)
    interps -= torch.abs(angle_vals)
    
    # Add new edges
    pos_interp_locs = torch.where(angle_vals > 1e-2)[0]
    pos_interps = angle_vals[pos_interp_locs]
    pos_edges = edge_index[:,pos_interp_locs]
    pos_selections = selections[pos_interp_locs] + 1
    pos_selections[torch.where(pos_selections>8)] = 1 # Account for wrap around
    
    neg_interp_locs = torch.where(angle_vals < -1e-2)[0]
    neg_interps = torch.abs(angle_vals[neg_interp_locs])
    neg_edges = edge_index[:,neg_interp_locs]
    neg_selections = selections[neg_interp_locs] - 1
    neg_selections[torch.where(neg_selections<1)] = 8 # Account for wrap around
    
    edge_index = torch.cat((edge_index,pos_edges,neg_edges),dim=1)
    selections = torch.cat((selections,pos_selections,neg_selections),dim=0)
    interps = torch.cat((interps,pos_interps,neg_interps),dim=0)
    
    return edge_index,selections,interps

def interpolateSelections_barycentric(edge_index,directions,d,vectorList=None):

    if vectorList is None:
        # Current Ordering
        # 4  3  2
        # 5  0  1
        # 6  7  8
        vectorList = torch.tensor([[1,0],[sqrt(2)/2,-sqrt(2)/2],[0,-1],[-sqrt(2)/2,-sqrt(2)/2],[-1,0],[-sqrt(2)/2,sqrt(2)/2],[0,1],[sqrt(2)/2,sqrt(2)/2]],dtype=torch.float).transpose(1,0).to(directions.device)

    # Normalize directions for simplicity of calculations
    dir_norm = torch.linalg.norm(directions,dim=1,keepdims=True)
    unit_directions = directions/dir_norm
    #locs = torch.where(dir_norm > 1)[0]
    #directions[locs] = directions[locs]/dir_norm[locs]
    
    values = torch.matmul(unit_directions,vectorList)
    best = torch.unsqueeze(torch.argmax(values,dim=1),1)
    #best_val = torch.take_along_dim(values,best,dim=1)
    
    # Look at both neighbors to see who is closer
    lower_val = torch.take_along_dim(values,(best-1) % 8,dim=1)
    upper_val = torch.take_along_dim(values,(best+1) % 8,dim=1)
    
    comp_vals = torch.cat((lower_val,upper_val),dim=1)
    
    second_best = torch.argmax(comp_vals,dim=1)
    #second_best_vals = torch.amax(comp_vals,dim=1)
    
    # Convert into uv cooridnates for barycentric interpolation calculation
    #     /|
    #    / |v
    #   /__|
    #    u 
    
    scaled_directions = torch.abs(directions/d)
    u = torch.amax(scaled_directions,dim=1)
    v = torch.amin(scaled_directions,dim=1)
    
    # Force coordinates to be within the triangle
    boundary_check = torch.where(u > d)
    v[boundary_check] /= u[boundary_check]
    u[boundary_check] = 1.0
    
    # Precalculated barycentric values from linear matrix solve
    I0 = 1 - u
    I1 = u - v
    I2 = v
    
    # Make first selections and proper interp_vals
    selections = best[:,0] + 1
    interp_vals = I1
    even_sels = torch.where(selections % 2 == 0)
    interp_vals[even_sels] = I2[even_sels] # Corners get different weights
    
    # Make new edges for the central selections
    central_edges = torch.clone(edge_index).to(edge_index.device)
    central_selections = torch.zeros_like(selections)
    central_interp_vals = I0
    
    # Make new edges for the last selection
    pos_locs = torch.where(second_best==1)[0]
    pos_edges = edge_index[:,pos_locs]
    pos_selections = selections[pos_locs] + 1
    pos_selections[torch.where(pos_selections>8)] = 1 #Account for wrap around
    pos_interp_vals = I1[pos_locs]
    even_sels = torch.where(pos_selections % 2 == 0)
    pos_interp_vals[even_sels] = I2[pos_locs][even_sels]
    
    neg_locs = torch.where(second_best==0)[0]
    neg_edges = edge_index[:,neg_locs]
    neg_selections = selections[neg_locs] - 1
    neg_selections[torch.where(neg_selections<1)] = 8 # Account for wrap around
    neg_interp_vals = I1[neg_locs]
    even_sels = torch.where(neg_selections % 2 == 0)
    neg_interp_vals[even_sels] = I2[neg_locs][even_sels]

    # Combine
    edge_index = torch.cat((edge_index,central_edges,pos_edges,neg_edges),dim=1)
    selections = torch.cat((selections,central_selections,pos_selections,neg_selections),dim=0)
    interp_vals = torch.cat((interp_vals,central_interp_vals,pos_interp_vals,neg_interp_vals),dim=0)
    
    
    
    # Account for edges to the same node
    same_locs = torch.where(edge_index[0] == edge_index[1])
    selections[same_locs] = 0
    interp_vals[same_locs] = 1
    # TODO make graph processing more efficient by removing 
    
    return edge_index,selections,interp_vals

def normalizeEdges(edge_index,selections,interps=None,kernel_norm=False):
    '''Given an edge_index and selections, normalize the edges for each node so that 
    aggregation of edges with interps = 1. If interps is given, use a weighted average.
    if kernel_norm = True, account for missing selections by increasing weight on other selections.'''
    
    N = torch.max(edge_index) + 1
    S = torch.max(selections) + 1
    
    total_weight = torch.zeros((N,S),dtype=torch.float).to(edge_index.device)
    
    if interps is None:
        interps = torch.ones(len(selections),dtype=torch.float).to(edge_index.device)
    
    # Aggregate all edges to determine normalizations per selection
    nodes = edge_index[0]
    #total_weight[nodes,selections] += interps
    total_weight.index_put_((nodes,selections),interps,accumulate=True)

    # Reassign interps accordingly
    if kernel_norm:
        row_totals = torch.sum(total_weight,dim=1)
        interps = interps * S/row_totals[nodes]
    else:
        norms = total_weight[nodes,selections]
        interps = interps/norms
    
    return interps

def simplifyGraph(edge_index,selections,edge_lengths):
    # Take the shortest edge for the set of the same selections on a given node
    num_edges = edge_index.shape[1]

    # Keep track of which nodes have been visited
    keep_edges = torch.zeros(num_edges,dtype=torch.bool).to(edge_index.device)

    previous_best_distance = 100000*torch.ones((torch.amax(edge_index)+1,torch.amax(selections)+1),dtype=torch.long).to(edge_index.device)
    previous_best_edge = -1*torch.ones((torch.amax(edge_index)+1,torch.amax(selections)+1),dtype=torch.long).to(edge_index.device)

    for i in range(num_edges):
        start_node = edge_index[0,i]
        #end_node = edge_index[1,i]
        selection = selections[i]
        distance = edge_lengths[i]

        if distance < previous_best_distance[start_node,selection]:
            previous_best_distance[start_node,selection] = distance
            keep_edges[i] = True

            prev = previous_best_edge[start_node,selection]
            if prev != -1:
                keep_edges[prev] = False

            previous_best_edge[start_node,selection] = i

    edge_index = edge_index[:,torch.where(keep_edges)[0]]
    selections = selections[torch.where(keep_edges)]

    return edge_index, selections
    
