import os
from PIL import Image
import time
from tqdm import tqdm
import numpy as np

# 경로 설정

path_dir = 'C:\cinemagraph_JPG\\'
file_list = os.listdir(path_dir)
file_list.sort()
jpg_list = []

print(len(file_list))

for i in range(225):
  path_jpg = path_dir + str(file_list[i]) + '\\'
  jpg_list.append(os.listdir(path_jpg))

#print(len(file_list))
#print(len(jpg_list))

# 이미지 경로 설정

path = []

for i in range(225):
    for j in range(195):
            try:
                path.append(path_dir + file_list[i] + '\\' + jpg_list[i][j])
            except:
                pass

print(len(path))

#print(len(path))

# 이미지 불러오기

im_list = []

for i in tqdm(path, desc='imageloading'):
    
    im = Image.open(i)
    im_resize = im.resize((128,128))
    im_list.append(np.asarray(im_resize))
    

#print(type(im_list))
#print(im_list[0])
#np.savetxt('./cinemagraphnumpy.txt',im_list)
np.save('./cinemagraphallnumpy.npy',im_list)