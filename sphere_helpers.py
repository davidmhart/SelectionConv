import py360convert
import torch
import graph_helpers as gh
from math import pi

def equirec2cubic(image,face_size=None):
    if face_size is None:
        cols = image.shape[1]
        face_size = cols//4
    return py360convert.e2c(image,face_size)

def cubic2equirec(cubemap,rows,cols):
    return py360convert.c2e(cubemap,rows,cols)

def equirec2spherical(rows, cols, device = 'cpu'):
    theta_steps = torch.linspace(0, 2*pi, cols+1).to(device)[:-1] # Avoid overlapping points
    phi_steps = torch.linspace(0, pi, rows+1).to(device)[:-1]
    theta, phi = torch.meshgrid(theta_steps, phi_steps,indexing='xy')
    return theta.flatten(),phi.flatten()

def spherical2equirec(theta,phi,rows,cols):
    x = theta*cols/(2*pi)
    y = phi*rows/pi
    return x,y

def spherical2xyz(theta,phi):
    x, y, z = torch.cos(theta) * torch.sin(phi), torch.cos(phi), torch.sin(theta) * torch.sin(phi),;
    return x,y,z


def buildCubemapEdges(horiz_nodes,top_nodes,bottom_nodes):

    rows,cols = top_nodes.shape
    
    src = []
    trg = []
    sel = []
    
    # horizontal
    gh.makeEdges(src,trg,sel,horiz_nodes[:,:-1], horiz_nodes[:,1:], 1)
    gh.makeEdges(src,trg,sel,horiz_nodes[:,-1], horiz_nodes[:,0], 1) # Wrap around
    # down right diagonal
    gh.makeEdges(src,trg,sel,horiz_nodes[:-1,:-1], horiz_nodes[1:,1:], 8)
    gh.makeEdges(src,trg,sel,horiz_nodes[:-1,-1], horiz_nodes[1:,0], 8) # Wrap Around
    # vertical
    gh.makeEdges(src,trg,sel,horiz_nodes[:-1,:], horiz_nodes[1:,:], 7)
    # down left diagonal
    gh.makeEdges(src,trg,sel,horiz_nodes[:-1,1:], horiz_nodes[1:,:-1], 6)
    gh.makeEdges(src,trg,sel,horiz_nodes[:-1,0], horiz_nodes[1:,-1], 6) # Wrap Around

    # center
    gh.makeEdges(src,trg,sel,horiz_nodes, horiz_nodes, 0, False)
    
    # Get the four triangular sections of the face and the diagonals
    for i in range(rows):
        for j in range(cols):
            # Downward triangle
            if i > j and i > (cols-1)-j:
                # horizontal
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i,j+1], 1, False)
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i,j-1], 5, False)

                if i == rows-1:
                    # Connect to 2nd cube view
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], horiz_nodes[0,cols+j+1], 8)
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], horiz_nodes[0,cols+j], 7)
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], horiz_nodes[0,cols+j-1], 6)
                else:
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i+1,j+1], 8, False) # down right diagonal
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i+1,j], 7, False) # vertical
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i+1,j-1], 6, False) # down left diagonal

                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i-1,j+1], 2, False) # up right diagnol
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i-1,j], 3, False) # vertical
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i-1,j-1], 4, False) # up left diagonal

            # Leftward triangle
            elif i > j and i < (cols-1)-j:
                # horizontal
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i+1,j], 1, False)
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i-1,j], 5, False)

                if j == 0:
                    # Connect to 1st cube view
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], horiz_nodes[0,i+1], 8)
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], horiz_nodes[0,i], 7)
                    if i == 0:
                        gh.makeEdges(src,trg,sel,top_nodes[i,j], horiz_nodes[0,-1], 6) # Wrap around to 4th view
                    else:
                        gh.makeEdges(src,trg,sel,top_nodes[i,j], horiz_nodes[0,i-1], 6)
                else:
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i+1,j-1], 8,False) # down right diagonal
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i,j-1], 7,False) # vertical
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i-1,j-1], 6,False) # down left diagonal

                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i+1,j+1], 2, False) # up right diagnol
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i,j+1], 3, False) # vertical
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i-1,j+1], 4, False) # up left diagonal

            # Upward triangle
            elif i < j and i < (cols-1)-j:
                # horizontal
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i,j-1], 1, False)
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i,j+1], 5, False)

                if i == 0:
                    # Connect to 4th cube view
                    if j == cols-1:
                        gh.makeEdges(src,trg,sel,top_nodes[i,j], horiz_nodes[0,0], 8) # Wrap around to 1st view
                    else:
                        gh.makeEdges(src,trg,sel,top_nodes[i,j], horiz_nodes[0,-1-j+1], 8)
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], horiz_nodes[0,-1-j], 7)
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], horiz_nodes[0,-1-j-1], 6)
                else:
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i-1,j-1], 8, False) # down right diagonal
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i-1,j], 7, False) # vertical
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i-1,j+1], 6, False) # # down left diagonal

                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i+1,j-1], 2, False) # up right diagnol
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i+1,j], 3, False) # vertical
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i+1,j+1], 4, False) # up left diagonal

            # Rightward triangle
            elif i < j and i > (cols-1)-j:
                # horizontal
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i-1,j], 1, False)
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i+1,j], 5, False)

                if j == cols-1:
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], horiz_nodes[0,3*cols-1-i+1], 8)
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], horiz_nodes[0,3*cols-1-i], 7)
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], horiz_nodes[0,3*cols-1-i-1], 6)
                else:
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i-1,j+1], 8, False) # down right diagonal
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i,j+1], 7, False) # vertical
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i+1,j+1], 6, False) # down left diagonal

                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i-1,j-1], 2, False) # up right diagnol
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i,j-1], 3, False) # vertical
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i+1,j-1], 4, False) # up left diagonal

            # Down Left Diagonol
            elif i == (cols-1)-j and j <= cols//2:
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i,j+1], 1, False)
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i,j+1], 2, False)
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i-1,j], 5, False) # Left in the direction of the next triangle
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i-1,j], 4, False)
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i-1,j+1], 3, False) # vertical in the direction of the center


                if i == rows - 1: # Corner
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], horiz_nodes[0,cols+1], 8)
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], horiz_nodes[0,cols], 7)
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], horiz_nodes[0,cols-1], 6)
                else:
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i+1,j], 8, False)
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i+1,j-1], 7, False)
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i,j-1], 6, False)

            # Up Left Diagonol
            elif i == j and j <= cols//2:
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i+1,j], 1, False)
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i+1,j], 2, False)
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i,j+1], 5, False) # Left in the direction of the next triangl
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i,j+1], 4, False)
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i+1,j+1], 3, False) # vertical in the direction of the center

                if i == 0: # Corner
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], horiz_nodes[0,1], 8)
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], horiz_nodes[0,0], 7)
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], horiz_nodes[0,-1], 6)
                else:
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i,j-1], 8, False)
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i-1,j-1], 7, False)
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i-1,j], 6, False)

            # Up Right Diagonol
            elif i == (cols-1)-j and j > cols//2:
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i,j-1], 1, False)
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i,j-1], 2, False)
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i+1,j], 5, False) # Left in the direction of the next triangle
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i+1,j], 4, False)
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i+1,j-1], 3, False) # vertical in the direction of the center

                if i == 0: # Corner
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], horiz_nodes[0,3*cols+1], 8)
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], horiz_nodes[0,3*cols], 7)
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], horiz_nodes[0,3*cols-1], 6)
                else:
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i-1,j], 8, False)
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i-1,j+1], 7, False)
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i,j+1], 6, False)

            # Down Right Diagonol
            elif i == j and j > cols//2:
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i-1,j], 1, False)
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i-1,j], 2, False)
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i,j-1], 5, False) # Left in the direction of the next triangle
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i,j-1], 4, False)
                gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i-1,j-1], 3, False) # vertical in the direction of the center

                if i == rows - 1: # Corner
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], horiz_nodes[0,2*cols+1], 8)
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], horiz_nodes[0,2*cols], 7)
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], horiz_nodes[0,2*cols-1], 6)
                else:
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i,j+1], 8, False)
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i+1,j+1], 7, False)
                    gh.makeEdges(src,trg,sel,top_nodes[i,j], top_nodes[i+1,j], 6, False)

    # centers
    gh.makeEdges(src,trg,sel,top_nodes, top_nodes, 0, False)

    
    # Get the four triangular sections of the face and the diagonals
    for i in range(rows):
        for j in range(cols):
            # Upward Triangle
            if i < j and i < (cols-1)-j:
                # horizontal
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i,j+1], 1, False)
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i,j-1], 5, False)

                if i == 0:
                    # Connect to 2nd cube view
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], horiz_nodes[-1,cols+j+1], 2)
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], horiz_nodes[-1,cols+j], 3)
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], horiz_nodes[-1,cols+j-1], 4)
                else:
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i-1,j+1], 2, False)
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i-1,j], 3, False)
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i-1,j-1], 4, False)

                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i+1,j+1], 8, False)
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i+1,j], 7, False)
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i+1,j-1], 6, False)

            # Leftward triangle
            elif i > j and i < (cols-1)-j:
                # horizontal
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i-1,j], 1, False)
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i+1,j], 5, False)

                if j == 0:
                    # Connect to 1st cube view
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], horiz_nodes[-1,cols-1-i+1], 2)
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], horiz_nodes[-1,cols-1-i], 3)
                    if i == 0:
                        gh.makeEdges(src,trg,sel,bottom_nodes[i,j], horiz_nodes[-1,-1], 4) # Wrap around to 4th view
                    else:
                        gh.makeEdges(src,trg,sel,bottom_nodes[i,j], horiz_nodes[-1,cols-1-i-1], 4)
                else:
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i-1,j-1], 2, False) # up right diagnol
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i,j-1], 3, False) # vertical
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i+1,j-1], 4, False) # up left diagonal

                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i-1,j+1], 8,False) # down right diagonal
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i,j+1], 7,False) # vertical
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i+1,j+1], 6,False) # down left diagonal


            # Downward triangle
            elif i > j and i > (cols-1)-j:
                # horizontal
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i,j-1], 1, False)
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i,j+1], 5, False)

                if i == rows-1:
                    # Connect to 4th cube view
                    if j == cols-1:
                        gh.makeEdges(src,trg,sel,bottom_nodes[i,j], horiz_nodes[-1,0], 2) # Wrap around to 1st view
                    else:
                        gh.makeEdges(src,trg,sel,bottom_nodes[i,j], horiz_nodes[-1,-1-j+1], 2)
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], horiz_nodes[-1,-1-j], 3)
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], horiz_nodes[-1,-1-j-1], 4)
                else:
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i+1,j-1], 2, False) # up right diagnol
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i+1,j], 3, False) # vertical
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i+1,j+1], 4, False) # up left diagonal

                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i-1,j-1], 8, False) # down right diagonal
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i-1,j], 7, False) # vertical
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i-1,j+1], 6, False) # # down left diagonal

            # Rightward triangle
            elif i < j and i > (cols-1)-j:
                # horizontal
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i+1,j], 1, False)
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i-1,j], 5, False)

                if j == cols-1:
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], horiz_nodes[-1,2*cols+i+1], 2)
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], horiz_nodes[-1,2*cols+i], 3)
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], horiz_nodes[-1,2*cols+i-1], 4)
                else:
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i+1,j+1], 2, False) # up right diagnol
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i,j+1], 3, False) # vertical
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i-1,j+1], 4, False) # up left diagonal

                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i+1,j-1], 8, False) # down right diagonal
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i,j-1], 7, False) # vertical
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i-1,j-1], 6, False) # down left diagonal


            # Down Left Diagonol
            elif i == (cols-1)-j and j <= cols//2:
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i,j+1], 5, False)
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i,j+1], 6, False)
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i-1,j], 1, False) # Right in the direction of the next triangle
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i-1,j], 8, False)
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i-1,j+1], 7, False) # vertical in the direction of the center

                if i == rows - 1: # Corner
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], horiz_nodes[-1,1], 2)
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], horiz_nodes[-1,0], 3)
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], horiz_nodes[-1,-1], 4)
                else:
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i+1,j], 2, False)
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i+1,j-1], 3, False)
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i,j-1], 4, False)

            # Up Left Diagonol
            elif i == j and j <= cols//2:
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i+1,j], 5, False)
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i+1,j], 6, False)
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i,j+1], 1, False) # Right in the direction of the next triangl
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i,j+1], 8, False)
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i+1,j+1], 7, False) # vertical in the direction of the center

                if i == 0: # Corner
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], horiz_nodes[-1,cols+1], 2)
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], horiz_nodes[-1,cols], 3)
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], horiz_nodes[-1,cols-1], 4)
                else:
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i,j-1], 2, False)
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i-1,j-1], 3, False)
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i-1,j], 4, False)

            # Up Right Diagonol
            elif i == (cols-1)-j and j > cols//2:
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i,j-1], 5, False)
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i,j-1], 6, False)
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i+1,j], 1, False) # Right in the direction of the next triangle
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i+1,j], 8, False)
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i+1,j-1], 7, False) # vertical in the direction of the center

                if i == 0: # Corner
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], horiz_nodes[-1,2*cols+1], 2)
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], horiz_nodes[-1,2*cols], 3)
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], horiz_nodes[-1,2*cols-1], 4)
                else:
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i-1,j], 2, False)
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i-1,j+1], 3, False)
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i,j+1], 4, False)

            # Down Right Diagonol
            elif i == j and j > cols//2:
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i-1,j], 5, False)
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i-1,j], 6, False)
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i,j-1], 1, False) # Right in the direction of the next triangle
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i,j-1], 8, False)
                gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i-1,j-1], 7, False) # vertical in the direction of the center

                if i == rows - 1: # Corner
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], horiz_nodes[-1,3*cols+1], 2)
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], horiz_nodes[-1,3*cols], 3)
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], horiz_nodes[-1,3*cols-1], 4)
                else:
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i,j+1], 2, False)
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i+1,j+1], 3, False)
                    gh.makeEdges(src,trg,sel,bottom_nodes[i,j], bottom_nodes[i+1,j], 4, False)

    # centers
    gh.makeEdges(src,trg,sel,bottom_nodes, bottom_nodes, 0, False)
    
    # Take the lists and turn them into true tensors
    edge_index = torch.row_stack((torch.tensor(src,dtype=torch.long),torch.tensor(trg,dtype=torch.long)))
    selections = torch.tensor(sel,dtype=torch.long)
    
    return edge_index, selections