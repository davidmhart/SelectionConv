from imageio import imread, imwrite
from py360convert import e2p

import matplotlib.pyplot as plt


#for name in ["style0.jpg","style1.jpg","style2.jpg","style3.jpg","style4.jpg","style6.jpg"]:
    
#name = "style1.jpg":

#content_fn = "paper_results/spherical_b/cube_map"+name
#save_fn = "paper_results/spherical_b/cubemap_view_"+name

content_fn = "test360b.jpg"
save_fn ="paper_results/spherical_b/test360b_view.jpg"

image = imread(content_fn)[:,:,:3]/255
rows,cols,ch = image.shape
view = e2p(image,(100,100),180,0,(rows,cols//2))
#view = e2p(image,(100,100),0,-90,(rows,cols//2))

#plt.imshow(view);plt.show()
imwrite(save_fn,view)