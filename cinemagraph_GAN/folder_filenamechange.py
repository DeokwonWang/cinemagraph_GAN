import os

path_dir = 'C:\cinemagraph_JPG\\'
file_list = os.listdir(path_dir)
trash = file_list[:136]

# print(trash)
j = 0

for i in trash:
  trash[j] = trash[j][4:]
  j += 1

print(trash)

i = 0
for foldername in os.listdir(path_dir)[:136]:
  os.rename(path_dir + foldername, path_dir + trash[i])
  i += 1