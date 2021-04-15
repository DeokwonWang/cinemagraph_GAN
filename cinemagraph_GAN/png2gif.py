import matplotlib.pyplot as plt
import numpy as np
import imageio
from PIL import Image
import matplotlib.image as mpimg
import os

path = [f"./gan_images/1/{i}" for i in os.listdir("./gan_images/1")]
path.sort()
paths = [Image.open(i) for i in path]
imageio.mimsave('./gan1.gif', paths, fps=5)