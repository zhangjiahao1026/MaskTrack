# MaskTrack
MaskTrack implement for [tianchi competition](https://tianchi.aliyun.com/competition/entrance/531797/introduction)
I hope to help those who have no ideas (^ _ ^)

## **score**

| J&F-Mean | J-Mean | J-Recall | J-Decay | F-Mean | F-Recall | F-Decay |
|:--------:|:------:|:--------:|:-------:|:------:|:--------:|:-------:|
|   0.7981 |  0.7793|0.8866    |0.0350   |0.8168  |0.8933    |0.0532   |

Reference to [omkar13/MaskTrack](https://github.com/omkar13/MaskTrack) , [paper](https://arxiv.org/abs/1612.02646v1)
1. **omkar13/MaskTrack** used Matlab to generate training data, which was too slow, so I implemented it in Python.
2. Use offline training without online fine-tuning

## Let's get started
We use one 'GeForce GTX 1080 Ti' cards with 11GB memory.

### **Environmental requirement**
1. pytorch
2. opencv
3. PIL

### **==> You need to change the path in path.py**

You just need to change **db_offline_train_root_dir** to dataset dir
```python
def db_offline_train_root_dir():
        return 'F:/vseg/dataset/media/tianchiyusai/tianchiyusai/'
```
There are three folders in **db_offline_train_root_dir()** which are **./Annotations  ./ImageSets ./JPEGImages**


### **==> Divide 50 videos into validation sets**

Run **split_train.py** to generate **train.txt** and **val.txt** in the current directory
Put **train.txt** and **val.txt** into **./ImageSets**

### **==> Generate training data**

Run **offline_data_generate.py**

### **==> Train**

Download the Deeplab Resnet 101 pretrained COCO model from [here](https://pan.baidu.com/s/1UaIXIOtWX0Z0xGJ1Av4yjg) (password xw09) and place it in 'pretrained/' folder.

Run **train_offline.py**

### **==> Test**

If you want to skip offline training,you can use [trained model](https://pan.baidu.com/s/1UaIXIOtWX0Z0xGJ1Av4yjg) (password xw09)

Run **online_test.py**


**Enjoy :)**