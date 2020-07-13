import os
import random
from PIL import Image
from path import Path

random.seed(11)
train_file_path = os.path.join(Path.db_offline_train_root_dir(),'ImageSets','train.txt')
with open(train_file_path) as f:
    videos = f.readlines()

random.shuffle(videos)
val_path = 'val.txt'
train_path = 'train.txt'
val_f = open(val_path,'w')
train_f = open(train_path,'w')
val_f.writelines(videos[-50:])
train_f.writelines(videos[:-50])

val_f.close()
train_f.close()

size_dict = {}
path = os.path.join(Path.db_offline_train_root_dir(),'Annotations')
for p in videos[-50:]:
    img_path = os.path.join(path,p[:6],'00000.png')
    img = Image.open(img_path)
    ss = '{}-{}'.format(img.size[0],img.size[1])
    if ss in size_dict:
        size_dict[ss]+=1
    else:
        size_dict[ss]=1
for k,v in size_dict.items():
    print(k,v)
""" 
1920-1080 7
1280-1080 1
1280-720 35
1104-622 2
720-1280 1
1280-960 1
1280-726 1
1280-644 1
1080-1920 1 """
    