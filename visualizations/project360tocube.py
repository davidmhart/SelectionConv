from imageio import imread, imwrite
from py360convert import e2c

import matplotlib.pyplot as plt

content_fn = "test360.jpg"
save_fn = "test360cube.jpg"
image = imread(content_fn)[:,:,:3]/255
rows,cols,ch = image.shape
cube = e2c(image,cols//4)

plt.imshow(cube);plt.show()
imwrite(save_fn,cube)