import numpy as np
from PIL import Image

x = np.load("./cinemagraphnumpy.npy")

color_img = x[0,:,:,:]
color_img = color_img.astype(np.uint8)
img = Image.fromarray(color_img)


img.save('realimg%.4d.png' % 3)