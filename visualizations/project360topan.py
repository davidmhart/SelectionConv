from imageio import imread, imwrite
from py360convert import e2c

import matplotlib.pyplot as plt

name = "style9.jpg"

content_fn = "../WorkingData/immenstadter_horn.jpg"
save_fn = "test360pan.jpg"
image = imread(content_fn)[:,:,:3]/255
rows,cols,ch = image.shape
pan = e2c(image,cols//4)
pan = pan[cols//4:2*cols//4]

plt.imshow(pan);plt.show()
imwrite(save_fn,pan)