#@author zjh

import numpy as np 
import cv2 
from PIL import Image
import random
import os
import datetime
class Masktrack_aug():
    def __init__(self,Davis_path=None):
        self.base_path = Davis_path
        self.annotation_path = os.path.join(self.base_path,'Annotations/')
        self.deformation_path = os.path.join(self.base_path,'Deformations/')
        self.gt_path = os.path.join(self.base_path,'Annotations_binary/')
        if not os.path.exists(self.deformation_path):
            os.makedirs(self.deformation_path)
        if not os.path.exists(self.gt_path):
            os.makedirs(self.gt_path)
        with open(os.path.join(self.base_path,'ImageSets','train.txt')) as f:
            trainset = [v[:-1] for v in f.readlines()]
        with open(os.path.join(self.base_path,'ImageSets','val.txt')) as f:
            valset = [v[:-1] for v in f.readlines()]  
        self.videos = trainset+valset  
        
        #self.videos = os.listdir(self.annotation_path)
        print('total {} videos'.format(len(self.videos)))
    def augment_image_and_mask(self,gt_arr,gt_path=None,affine_transformation_path=None, non_rigid_deform_path=None):
        #gt_arr shape (H,W) and binary(0,1)

        # let us do non-rigid deformation
        N = 5
        Delta = 0.05
        H,W = gt_arr.shape
        #get the target boundary
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        boundary = cv2.dilate(gt_arr, kernel)-gt_arr
        boundindex = np.where(boundary==1)
        num_index = boundindex[0].shape[0]
        if num_index>N:
            maxH,minH = max(boundindex[0]),min(boundindex[0])
            tarH = maxH - minH
            maxW,minW = max(boundindex[1]),min(boundindex[1])
            tarW = maxW - minW

            # thin plate spline coord num    
            randindex = [random.randint(0,num_index-1) for _ in range(N)]
            sourcepoints=[]
            targetpoints = []
            for i in range(N):
                sourcepoints.append((boundindex[1][randindex[i]],boundindex[0][randindex[i]]))
                x = boundindex[1][randindex[i]]+int(random.uniform(-Delta,Delta)*tarW)
                y = boundindex[0][randindex[i]]+int(random.uniform(-Delta,Delta)*tarH)
                targetpoints.append((x,y))
        
            sourceshape = np.array(sourcepoints,np.int32)
            sourceshape=sourceshape.reshape(1,-1,2)
            targetshape = np.array(targetpoints,np.int32)
            targetshape=targetshape.reshape(1,-1,2)

            matches =[]
            for i in range(0,N):
                matches.append(cv2.DMatch(i,i,0))
            tps= cv2.createThinPlateSplineShapeTransformer()
            tps.estimateTransformation(targetshape, sourceshape,matches)
            no_grid_img=tps.warpImage(gt_arr)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            no_grid_img = cv2.dilate(no_grid_img,kernel)
            #for p in sourcepoints:
            #    cv2.circle(gt_arr,p,2,(2),2)   
            #for p in targetpoints:
            #    cv2.circle(no_grid_img,p,2,(2),2)
            gt_out = gt_arr*255
            no_grid_img = no_grid_img*255
            gt_out = Image.fromarray(gt_out)
            no_grid_out = Image.fromarray(no_grid_img)
            
            #let us do affine transformation

            scale=0.98
            randScale = random.uniform(scale,1/scale)
            M = cv2.getRotationMatrix2D(((maxH+minH)*0.5, (maxW-minW)*0.5), 0, randScale)
            
            dx = round(random.uniform(-0.05,0.05)*tarW)
            dy = round(random.uniform(-0.05,0.05)*tarH)
            M[0,2]+=dx
            M[1,2]+=dy
            affine_out = cv2.warpAffine(gt_arr, M, (W, H))*255
            affine_out = Image.fromarray(affine_out)
        else:
            gt_out = Image.fromarray(gt_arr*255)
            no_grid_out = Image.fromarray(gt_arr*255)
            affine_out = Image.fromarray(gt_arr*255)

        #save gt and none_rigid_deform and affine_tran
        gt_out.save(gt_path)
        no_grid_out.save(non_rigid_deform_path)
        affine_out.save(affine_transformation_path)

    def script(self):
        start_time = datetime.datetime.now()
        for kk,video in enumerate(self.videos):
            nowtime = datetime.datetime.now()
            remain = (nowtime-start_time)*(len(self.videos)-kk-1)/(kk+1)
            print('{} remain {} {}-{}'.format(nowtime,remain,kk+1,video))
            mask_gt_path = os.path.join(self.annotation_path,video)
            label_folder_path = os.path.join(self.gt_path,video)
            deform_folder_path = os.path.join(self.deformation_path,video)
            if  os.path.exists(label_folder_path):
                continue
            os.makedirs(label_folder_path)
            if not os.path.exists(deform_folder_path):
                os.makedirs(deform_folder_path)
            frames = os.listdir(mask_gt_path)
            no_objects = 1
            for k,frame in enumerate(frames):
                #print('Deal with {}-{}'.format(video,frame))
                frame_path = os.path.join(mask_gt_path,frame)
                frame_gt_image = Image.open(frame_path)
                frame_gt_image = np.array(frame_gt_image)
                frame_index = frame[:-4]
                if k==0:
                    no_objects = frame_gt_image.max()
                for object_id in range(1,no_objects+1):
                    temp = frame_gt_image.copy()
                    m1 = temp==object_id
                    m0 = temp!=object_id
                    temp[m1]=1
                    temp[m0]=0

                    gt_path = os.path.join(label_folder_path,frame_index+'_'+str(object_id)+'.png')
                    aff_path = os.path.join(deform_folder_path,frame_index+'_'+str(object_id)+'_d1.png')
                    non_path = os.path.join(deform_folder_path,frame_index+'_'+str(object_id)+'_d2.png')
                    self.augment_image_and_mask(temp,gt_path=gt_path,
                                                affine_transformation_path=aff_path,
                                                non_rigid_deform_path=non_path)       

        
if __name__ == '__main__':
    from path import Path
    masktrack_aug = Masktrack_aug(Davis_path=Path.db_offline_train_root_dir())
    masktrack_aug.script()

