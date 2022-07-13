import matplotlib.pyplot as plt
import numpy as np
from numpy import expand_dims as unsqueeze

from matplotlib.path import Path
import graph_helpers as gh
from mesh_helpers import getUVs

from scipy.ndimage import binary_erosion, binary_dilation

import torch

def texture2Points3D(mask,mesh):
    
    vertices = mesh.vertices
    uvs = getUVs(mesh)
    faces = mesh.faces
    face_normals = mesh.face_normals

    rows,cols = mask.shape

    # Get the uv of each pixel
    pos_rows = np.linspace(0,1,rows)
    pos_cols = np.linspace(0,1,cols)
    xv, yv = np.meshgrid(pos_cols,pos_rows)
    yv = 1-yv #UV has positive y-axis
    pos_all = np.stack((xv,yv),axis=-1)
    pos_masked = pos_all[np.where(mask)]
    pos = np.reshape(pos_masked,(-1,2))

    # Get Image Positions
    row_space = np.arange(rows)
    col_space = np.arange(cols)
    col_image,row_image = np.meshgrid(col_space,row_space)
    im_pos_all = np.stack((row_image,col_image),axis=-1)
    im_pos_masked = im_pos_all[np.where(mask)]
    im_pos = np.reshape(im_pos_masked,(-1,2))

    # Find the faces to which each pixel belongs
    face_nums = np.zeros(len(pos)).astype(np.int32)
    locs = np.arange(len(pos))

    for f_num, face in enumerate(faces):

        uv = uvs[face]

        # Bounding box test
        xmin = np.amin(uv)
        xmax = np.amax(uv)
        ymin = np.amin(uv)
        ymax = np.amax(uv)

        possible = np.where(np.logical_and(np.logical_and(pos[:,0] >= xmin, pos[:,0] <= xmax), np.logical_and(pos[:,1] >= ymin, pos[:,1] <= ymax)))

        # Use path library to test if inside the polygon
        poly_path = Path(uv)

        valid = poly_path.contains_points(pos[possible])

        final_locs = locs[possible][valid]

        face_nums[final_locs] = f_num

    #print(face_nums)

    # Approximate each pixels location in 3D space
    pos3D = np.zeros((len(pos),3))
    normals = np.zeros((len(pos),3))

    for pos_num, p in enumerate(pos):

        face = faces[face_nums[pos_num]]
        uv = uvs[face]

        # Calculate the barycentric coordinates
        A = np.matrix([[uv[0,0],uv[1,0],uv[2,0]],
                       [uv[0,1],uv[1,1],uv[2,1]],
                       [1,1,1]])
        b = np.matrix([[p[0]],[p[1]],[1]])

        bary = np.linalg.inv(A)*b

        # Calculate the 3D coordinate from the 2D bary coordinate
        verts = vertices[face]
        pos3D[pos_num] = bary[0,0]*verts[0] + bary[1,0]*verts[1] + bary[2,0]*verts[2]

        # Look up the face normal
        normals[pos_num] = face_normals[face_nums[pos_num]]
        
    pos3D = torch.tensor(pos3D,dtype=torch.float)
    normals = torch.tensor(normals,dtype=torch.float)
        
    return pos3D, normals
    

def seperateTexture(mesh, rows, cols, return_lookup = False):

    vertices = mesh.vertices
    faces = mesh.faces
    uvs = getUVs(mesh)
    
    x_pos = np.linspace(0,1,cols)
    y_pos = np.linspace(0,1,rows)
    xv, yv = np.meshgrid(x_pos, y_pos)
    yv = 1-yv # uv coordinates use bottom left origin
    pos = np.stack((xv,yv),axis=2)

    mask = np.zeros((rows,cols))

    # Boundary edges are only referenced by one triangle face
    count = np.zeros((len(uvs),len(uvs)))
    for face in faces:
        count[face[0],face[1]] += 1
        count[face[1],face[2]] += 1
        count[face[2],face[0]] += 1
        
    # TODO Switch to sparse matrix or hashmap for efficiency

    boundary = np.where(count + count.T == 1)
    boundary = np.stack((boundary[0], boundary[1]),axis=1)

    # Make edges bidirectional (i.e. remove duplicate entries)
    boundary = boundary[np.where(boundary[:,0] <= boundary[:,1])[0]]


    # Get all closed loops
    cycles, innerCycles = _findClosedLoops(boundary)
    
    # Account for internal loops
    cycles, innerCycles = _findInnerCycles(cycles,uvs, innerCycles)

    # Remove empty list
    remove_list = []
    for c, cycle in enumerate(innerCycles):
        if len(cycle) == 0:
            remove_list.append(c)
            
    for index in sorted(remove_list, reverse=True):
        del innerCycles[index]
        
    # Mask out all visible pixels
    for cycle in cycles:
    
        # All first entries represent the outside points of the boundaries
        r = uvs[cycle[:,0]]
        
        # Bounding Box Test
        xmin = np.amin(r[:,0])
        xmax = np.amax(r[:,0])
        ymin = np.amin(r[:,1])
        ymax = np.amax(r[:,1])

        possible = np.where(np.logical_and(np.logical_and(pos[:,:,0] >= xmin, pos[:,:,0] <= xmax), np.logical_and(pos[:,:,1] >= ymin, pos[:,:,1] <= ymax)))
        
        # Use path library to test if inside the polygon
        poly_path = Path(r)
        
        valid = poly_path.contains_points(pos[possible])
        
        mask[possible] = np.logical_or(mask[possible],valid)

    for cycle in innerCycles:
        # Unmask all internal pixels
        
        # All first entries represent the outside points of the boundaries
        r = uvs[cycle[:,0]]
        
        # Bounding Box Test
        xmin = np.amin(r[:,0])
        xmax = np.amax(r[:,0])
        ymin = np.amin(r[:,1])
        ymax = np.amax(r[:,1])

        possible = np.where(np.logical_and(np.logical_and(pos[:,:,0] >= xmin, pos[:,:,0] <= xmax), np.logical_and(pos[:,:,1] >= ymin, pos[:,:,1] <= ymax)))
        
        # Use path library to test if inside the polygon
        poly_path = Path(r)
        
        valid = poly_path.contains_points(pos[possible])

        # Remove pixels that are in inner loops
        valid = np.logical_not(valid)
        mask[possible] = np.logical_and(mask[possible],valid)
        
    cycles = cycles + innerCycles
     
    mask = mask.astype(np.bool)
        
    if return_lookup:
        
        # Make Lookup Table that identifies which edges are the same in each subimages
        lookup = {}
        for m in range(len(cycles)):
            edges = cycles[m]

            for i,edge in enumerate(edges):
                start = unsqueeze(vertices[edge[0]],axis=0)
                end = unsqueeze(vertices[edge[1]],axis=0)

                match_never_found = True

                for n in range(len(cycles)):
                    if m == n:
                        # Make sure to not include the current edge
                        if i == len(edges)-1:
                            edges2 = edges[:i]
                        else:
                            edges2 = np.concatenate((edges[:i],edges[i+1:]),axis=0)

                        forward = np.logical_and(np.isclose(vertices[edges2[:,0]],start), np.isclose(vertices[edges2[:,1]], end))

                        # Make sure all 3 values are equal
                        forward = np.prod(forward,axis=1)

                        matches = np.where(forward)[0]

                        if len(matches) > 0:
                            match = matches[0]

                            if match < i:
                                lookup[m,i] = n,match,"forward"
                            else:
                                lookup[m,i] = n,match + 1,"forward"
                            match_never_found=False
                            break

                        reverse = np.logical_and(np.isclose(vertices[edges2[:,1]],start), np.isclose(vertices[edges2[:,0]], end))
                        reverse = np.prod(reverse,axis=1)
                        matches = np.where(reverse)[0]

                        if len(matches) > 0:
                            match = matches[0]

                            if match < i:
                                lookup[m,i] = n,match,"reverse"
                            else:
                                lookup[m,i] = n,match + 1,"reverse"
                            match_never_found=False
                            break

                    else:
                        edges2 = cycles[n]
                        forward = np.logical_and(np.isclose(vertices[edges2[:,0]],start), np.isclose(vertices[edges2[:,1]], end))

                        # Make sure all 3 values are equal
                        forward = np.prod(forward,axis=1)
                        matches = np.where(forward)[0]


                        if len(matches) > 0:
                            match = matches[0]
                            lookup[m,i] = n,match,"forward"
                            match_never_found=False
                            break

                        reverse = np.logical_and(np.isclose(vertices[edges2[:,1]],start), np.isclose(vertices[edges2[:,0]], end))                    
                        reverse = np.prod(reverse,axis=1)
                        matches = np.where(reverse)[0]

                        if len(matches) > 0:
                            match = matches[0]
                            lookup[m,i] = n,match,"reverse"
                            match_never_found=False
                            break


                if match_never_found:
                    print("Match Not Found!",m,i)

        # TODO Use hashmap to make more efficient
        #plotBoundaryPairs(lookup,cycles,uvs)

        return mask, cycles, lookup
    
    else:
        return mask, cycles

def buildTextureEdges(mask,boundaries,lookup,mesh,rows,cols):
    
    vertices = mesh.vertices
    faces = mesh.faces
    uvs = getUVs(mesh)
    
    # Get Image Positions
    row_space = np.arange(rows)
    col_space = np.arange(cols)
    col_image,row_image = np.meshgrid(col_space,row_space,indexing='xy')
    im_pos = np.reshape(np.stack((row_image,col_image),axis=-1),(rows*cols,2))
    
    # Canonical Space Coordinates
    pos_rows = np.linspace(0,1,rows)
    pos_cols = np.linspace(0,1,cols)
    xv, yv = np.meshgrid(pos_cols,pos_rows)
    yv = 1-yv #UV has positive y-axis
    pos = np.reshape(np.stack((xv,yv),axis=-1),(rows*cols,2))

    # Precompute border pixels for later computation
    mask_border = mask ^ binary_erosion(mask,iterations=1)
    node_border = np.where(np.reshape(mask_border,(rows*cols)))
    im_pos_border = im_pos[node_border]
    pos_border = pos[node_border]

    # Mask out nodes
    node_mask = np.where(np.reshape(mask,(rows*cols)))
    im_pos = im_pos[node_mask]
    pos = pos[node_mask]
    
    # Generate graph for regular image groups
    pos2D = gh.convertImPos(torch.tensor(im_pos,dtype=torch.long),flip_y=False)
    edge_index = gh.grid2Edges(pos2D)
    directions = pos2D[edge_index[1]] - pos2D[edge_index[0]]
    selections = gh.edges2Selections(edge_index,directions,interpolated=False,y_down=False)

    ######## Intergroup connections ######
    
    eps = max(1/rows,1/cols) # Epsilon for pixels on border based on rows and cols

    inter_edges = []
    inter_selections = []

    # For each boundary edge
    for m in range(len(boundaries)):
        for j in range(len(boundaries[m])):
            # Find it's paired edge
            try:

                n,k,orientation = lookup[m,j]

                # Find all pixels on the edge
                edge = boundaries[m][j]
                uv1 = uvs[edge[0]]
                uv2 = uvs[edge[1]]

                #plt.scatter(uv1[0],uv1[1],c="r")
                #plt.scatter(uv2[0],uv2[1],c="r")

                # Bounding Box Reject
                xmin = min(uv1[0],uv2[0]) - 1/cols
                xmax = max(uv1[0],uv2[0]) + 1/cols
                ymin = min(uv1[1],uv2[1]) - 1/rows
                ymax = max(uv1[1],uv2[1]) + 1/rows
                possible = np.where(np.logical_and(np.logical_and(pos_border[:,0] >= xmin, pos_border[:,0] <= xmax), np.logical_and(pos_border[:,1] >= ymin, pos_border[:,1] <= ymax)))

                # Formula for the distance of a point from a line
                dist = ((uv2[0] - uv1[0])*(uv1[1] - pos_border[possible][:,1]) - (uv2[1] - uv1[1])*(uv1[0] - pos_border[possible][:,0]))/np.sqrt((uv2[0] - uv1[0])**2 + (uv2[1] - uv1[1])**2)

                valid = np.where(dist < eps)

                pos_current = pos_border[possible][valid]
                im_pos_current = im_pos_border[possible][valid]

                # Test all connections for each node on the edge
                #for c,im_pos_c in enumerate(im_pos_current):
                for im_pos_c in im_pos_current:
                    node_num = np.where(np.logical_and(im_pos_c[0] == im_pos[:,0],im_pos_c[1] == im_pos[:,1]))[0][0]

                    #print(node_num,im_pos_k,im_pos[node_num])
                    #print(pos[node_num],pos_current[c])

                    edge1 = boundaries[m][j]
                    edge2 = boundaries[n][k]

                    # Reverse edge2 if vertices are listed in reverse order from edge1
                    if orientation == "reverse":
                        edge2 = edge2[::-1]

                    # Make appropriate connections in all 8 directions
                    _checkDirection([0,1],1,node_num,im_pos_c,inter_edges,inter_selections,mask,pos,edge1,edge2,uvs,im_pos,rows,cols)
                    _checkDirection([1,1],2,node_num,im_pos_c,inter_edges,inter_selections,mask,pos,edge1,edge2,uvs,im_pos,rows,cols)
                    _checkDirection([1,0],3,node_num,im_pos_c,inter_edges,inter_selections,mask,pos,edge1,edge2,uvs,im_pos,rows,cols)
                    _checkDirection([1,-1],4,node_num,im_pos_c,inter_edges,inter_selections,mask,pos,edge1,edge2,uvs,im_pos,rows,cols)
                    _checkDirection([0,-1],5,node_num,im_pos_c,inter_edges,inter_selections,mask,pos,edge1,edge2,uvs,im_pos,rows,cols)
                    _checkDirection([-1,-1],6,node_num,im_pos_c,inter_edges,inter_selections,mask,pos,edge1,edge2,uvs,im_pos,rows,cols)
                    _checkDirection([-1,0],7,node_num,im_pos_c,inter_edges,inter_selections,mask,pos,edge1,edge2,uvs,im_pos,rows,cols)
                    _checkDirection([-1,1],8,node_num,im_pos_c,inter_edges,inter_selections,mask,pos,edge1,edge2,uvs,im_pos,rows,cols)
            except KeyError:
                pass

    # Append all intergroup edges
    #print("Inter Edges:", len(inter_edges))
    inter_edge_index = torch.tensor(inter_edges,dtype=torch.long).transpose(1,0)
    edge_index = torch.cat((edge_index,inter_edge_index),dim=1)

    # Append all interedge selections
    inter_selections = torch.tensor(inter_selections,dtype=torch.long)
    selections = torch.cat((selections,inter_selections))
    
    return edge_index, selections


def textureDilation(canvas,mask=None,iterations=1):

    # First, generate mask to find boundary pixels
    if mask is None:
        rows,cols,_ = canvas.shape
        mask = np.zeros((rows,cols)).astype(np.bool)
        mask[np.where(np.sum(canvas,axis=2) > 0)] = True

    #plt.imshow(mask);plt.show()
    #print(mask.dtype)
    
    # Next, iteratively fill border pixels
    for i in range(iterations):

        border = binary_dilation(mask) ^ mask

        #plt.imshow(border);plt.show()

        border_pixs = np.where(border)

        reference = np.copy(canvas)

        for k in range(len(border_pixs[0])):

            r = border_pixs[0][k]
            c = border_pixs[1][k]

            #plt.imshow(reference[r-1:r+2,c-1:c+2]);plt.show()
            #plt.imshow(mask[r-1:r+2,c-1:c+2]);plt.show()

            # Normalize based on the number of touching border pixels
            # canvas[r,c] = np.sum(reference[r-1:r+2,c-1:c+2],axis=(0,1))/np.sum(mask[r-1:r+2,c-1:c+2])

            # Normalize based on the number of touching border pixels, weigh diagonols less
            values = reference[r-1:r+2,c-1:c+2]
            valid = mask[r-1:r+2,c-1:c+2].astype(np.float32)
            values = np.multiply(values,np.expand_dims(valid,axis=2))
            values[::2,::2] *= .707107 # Multiply diagonols by 1/sqrt(2) to account for euclidean distance
            valid[::2,::2] *= .707107
            #plt.imshow(values);plt.show()
            #plt.imshow(valid);plt.show()
            canvas[r,c] = np.sum(values,axis=(0,1))/np.sum(valid)

            #plt.imshow(canvas[r-1:r+2,c-1:c+2]);plt.show()

        mask = mask | border

    return canvas,mask

def _findClosedLoops(edges):
    # Input should be N,2 numpy array
    # Assumes bidirectional edges

    #print(edges.shape)

    cycles = []
    cycle_found = False
    explored = set()
    
    inner_cycles = []
    
    for start in range(edges.shape[0]):
        cycle = []
    
        if start in explored:
            # Loop already found or not possible
            continue

        explored.add(start)
        cycle.append(edges[start])
        
        start_node = edges[start,0]
        head = edges[start,1]
        
        prev_heads = [head] # Used to check for inner loops
        
        #print(start_node,head)
        
        connection_found = True
        
        while connection_found:
            
            #print(head)
            
            connection_found = False
            
            for i in range(edges.shape[0]):
                if i in explored:
                    # Already visited
                    #print("Visited:",i)
                    continue
            
                if edges[i,0] == head:
                    explored.add(i)
                    cycle.append(edges[i])
                    head = edges[i,1]
                    connection_found = True
                    break
                
                if edges[i,1] == head:
                    explored.add(i)
                    cycle.append(edges[i,::-1])
                    head = edges[i,0]
                    connection_found = True
                    break
                
            if head == start_node:
                cycle = np.array(cycle)
                cycles.append(cycle)
                break
            
            # Found internal loop
            if head in prev_heads:
                index = prev_heads.index(head)
                inner_cycle = cycle[index+1:]
                inner_cycles.append(np.array(inner_cycle))
                cycle = cycle[:index+1]
                
                
                
            prev_heads.append(head)
            
    return cycles, inner_cycles
    
def _findInnerCycles(cycles,vertices,innerCycles = None):
    
    remove_list = []
    
    if innerCycles is None:
       innerCylces = []
    
    for i in range(len(cycles)):
    
        if i in remove_list:
            continue
    
        outside = vertices[cycles[i][:,0]]
        poly_path = Path(outside)
    
        for j in range(len(cycles)):
        
            if i == j:
                continue
        
            if j in remove_list:
                continue
            
            inside = vertices[cycles[j][:,0]]  
            tests = poly_path.contains_points(inside)
            
            # 75% of points lie in the other
            if np.mean(tests)>=.75:
                remove_list.append(j)
    
    for index in sorted(remove_list, reverse=True):
        innerCycles.append(cycles[index])
        del cycles[index]
    
    #print(len(cycles))
    
    return cycles, innerCycles
           
def _makeNewConnection(index,pos,direction,edge1,edge2,uvs,im_pos,rows,cols):

    p = pos[index]
    r = direction
    
    e1a = uvs[edge1[0]]
    e1b = uvs[edge1[1]]
    e1 = e1b-e1a

    A = np.matrix([[e1[0],-r[0]],[e1[1],-r[1]]])

    if np.abs(np.linalg.det(A)) < 1e-10:
        print("Found Singular Matrix")
        return None

    # Determine the intersection with the initial edge
    intersection = np.matmul(np.linalg.inv(A),np.matrix([[p[0]-e1a[0]],[p[1]-e1a[1]]]))

    a = intersection[0,0]
    c = intersection[1,0]

    if a < 0 or a > 1 or c < 0:
        #print("Im_check_failed?")
        #if im_pos[index][0] == 138 and im_pos[index][1] == 70:
        #    print("Got Here!")
        #    print(a,c)
        #    breakpoint()

        return None

    else:
        #print(intersection)

        e2a = uvs[edge2[0]]
        e2b = uvs[edge2[1]]
        e2 = e2b - e2a
        b = a #*norm(e2)/norm(e1)
        d = (np.linalg.norm(r)-c)* np.linalg.norm(e2)/np.linalg.norm(e1)

        e1u = e1/np.linalg.norm(e1)
        e2u = e2/np.linalg.norm(e2)

        R = np.matrix([[e1u[0]*e2u[0]+e1u[1]*e2u[1],e2u[0]*e1u[1]-e1u[0]*e2u[1]],[e1u[0]*e2u[1]-e2u[0]*e1u[1],e1u[0]*e2u[0]+e1u[1]*e2u[1]]])

        rm = np.matrix(r)

        new_p = np.array(b*e2 + e2a + d*np.matmul(R,rm.T).T)[0]

        #print(new_p)

        end_row = int(np.round((1-new_p[1]) * rows))
        end_col = int(np.round(new_p[0] * cols))

        end_node = np.where(np.logical_and(end_row == im_pos[:,0],end_col == im_pos[:,1]))[0]


        #if im_pos[index][0] == 144 and im_pos[index][1] == 237:
            #print("Got through!")
            #breakpoint()


        if len(end_node) == 0 :

            # Try going in the opposite direction
            d = -d

            new_p = np.array(b*e2 + e2a + d*np.matmul(R,rm.T).T)[0]
            end_row = int(np.round((1-new_p[1]) * rows))
            end_col = int(np.round(new_p[0] * cols))

            end_node = np.where(np.logical_and(end_row == im_pos[:,0],end_col == im_pos[:,1]))[0]

            if len(end_node) == 0 :
                return None

        return [index,end_node[0]]

def _checkDirection(direction,selection,node_num,im_pos_c,inter_edges,inter_selections,mask,pos,edge1,edge2,uvs,im_pos,rows,cols):

    # Check the current node
    if mask[im_pos_c[0],im_pos_c[1]]:

        # Check the node it is going towards
        im_pos_check = im_pos_c + np.array(direction)
        if not mask[im_pos_check[0],im_pos_check[1]]:

            # Make New Connection
            pos_direction = np.array([direction[1]/cols,-direction[0]/rows])
            new_edge = _makeNewConnection(node_num,pos,pos_direction,edge1,edge2,uvs,im_pos,rows,cols)
            if not new_edge is None:
                inter_edges.append(new_edge)
                inter_selections.append(selection)

    
def plotBoundary(image,verts,boundary,box_bounds=None):
    
    rows = image.shape[0]
    cols = image.shape[1]
    
    if box_bounds == None:
        y_start = 0
        y_stop = cols
        x_start = 0
        x_stop = rows
    else:
        y_start = box_bounds[0].start
        y_stop = box_bounds[0].stop
        x_start = box_bounds[1].start
        x_stop = box_bounds[1].stop
    
    print(verts[boundary[:,0],0]*cols - x_start)
    
    plt.imshow(image[y_start:y_stop,x_start:x_stop])
    plt.plot([verts[boundary[:,0],0]*cols-x_start,verts[boundary[:,1],0]*cols-x_start],\
              [(1-verts[boundary[:,0],1])*rows-y_start,(1-verts[boundary[:,1],1])*rows-y_start], 'r', lw=2)
    plt.show()

def plotAllBoundaries(image,verts,boundaries):
    
    rows = image.shape[0]
    cols = image.shape[1]

    #print(verts[boundary[:,0],0]*cols - x_start)
    
    plt.imshow(image)
    
    for boundary in boundaries:
        plt.plot([verts[boundary[:,0],0]*cols,verts[boundary[:,1],0]*cols],\
              [(1-verts[boundary[:,0],1])*rows,(1-verts[boundary[:,1],1])*rows], 'r', lw=2)
    
    plt.show()
    
    
def plotBoundaryPairs(lookup,boundaries,verts):
    
    edges=[]
    colors=[]
    visited = {}
    
    for key in lookup:
        # Check that both edges reference each other
        
        value = lookup[key]
        
        try:
            if visited[key]:
                pass
        except:
            try:
                comp = lookup[(value[0],value[1])]
                
                if (comp[0],comp[1]) == (key[0],key[1]):
                
                    e1 = boundaries[key[0]][key[1]]
                    e2 = boundaries[value[0]][value[1]]
                    
                    edges.append(e1)
                    edges.append(e2)
                    color = tuple(np.random.rand(3))
                    colors.append(color)
                    colors.append(color)
                    
                    visited[key] = True
                    visited[value] = True
                    
            except KeyError:
                print("No Match for",key,value)

    #plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(edges)))))

    edges = np.array(edges)
    print(len(edges))
    #colors = np.array(colors)

    plt.axis([0,1,0,1])
#    plt.ion()

    for i in range(0,len(edges),2):
        plt.plot([verts[edges[i,0],0],verts[edges[i,1],0]],\
              [(1-verts[edges[i,0],1]),(1-verts[edges[i,1],1])], color=colors[i], lw=2)
        plt.plot([verts[edges[i+1,0],0],verts[edges[i+1,1],0]],\
              [(1-verts[edges[i+1,0],1]),(1-verts[edges[i+1,1],1])], color=colors[i], lw=2)
#        plt.pause(0.5)
    
    #plt.plot([verts[edges[:,0],0],verts[edges[:,1],0]],\
    #          [(1-verts[edges[:,0],1]),(1-verts[edges[:,1],1])], color=colors, lw=2)
    plt.show()


