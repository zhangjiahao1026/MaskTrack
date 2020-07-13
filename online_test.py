import torch
import numpy
import cv2
import collections
import deeplab_resnet
from PIL import Image
import numpy as np 
from torch.autograd import Variable
import os
import datetime
import shutil
from path import Path

class Online_test():
    def __init__(self,use_cuda=True,inputRes=(480,854),model_path=None,base_path=None):
        self.use_cuda = use_cuda
        self.model_path = model_path
        self.base_path = base_path
        self.large_neg = -1e6
        self.inputRes = inputRes
    def get_frame_name(self,n):
        s = '00000'+str(n)
        return s[-5:]
    def resize_and_padding(self,img,df,inputRes=None):
        n_h=-1
        n_w=-1
        if inputRes==None:
            return n_h,n_w,img,df
        w_div_h = float(inputRes[1])/inputRes[0]
        im_w_h = float(df.shape[1])/df.shape[0]
        if abs(w_div_h-im_w_h)<0.1:
            img = cv2.resize(img,(inputRes[1],inputRes[0]),interpolation=cv2.INTER_CUBIC)
            df = cv2.resize(df,(inputRes[1],inputRes[0]),interpolation=cv2.INTER_NEAREST)
            return n_h,n_w,img,df
        
        if im_w_h > w_div_h:
            scale = inputRes[1]/df.shape[1]
            conv_h = int(df.shape[0]*scale)
            n_h = conv_h
            n_w = inputRes[1]
        else:
            scale = inputRes[0]/df.shape[0]
            conv_w = int(df.shape[1]*scale)
            n_h = inputRes[0]
            n_w = conv_w
        img_s = cv2.resize(img,(n_w,n_h),interpolation=cv2.INTER_CUBIC)
        df_s = cv2.resize(df,(n_w,n_h),interpolation=cv2.INTER_NEAREST)

        img = np.zeros((inputRes[0],inputRes[1],3),dtype=np.uint8)
        df = np.zeros((inputRes[0],inputRes[1]),dtype=np.uint8)

        img[:n_h,:n_w,:] = img_s
        df[:n_h,:n_w] = df_s
        return n_h,n_w,img,df
    def test_without_train(self,net=None,first_mask_path=None,img_dir=None,save_dir=None,use_cuda=True):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        Nframes = len([int(s[:-4]) for s in os.listdir(img_dir)])
        N_object = 1
        palette = None
        meanval = (104.00699, 116.66877, 122.67892)
        for i_frame in range(1,Nframes):
            if i_frame==1:
                mask_path = first_mask_path
            else:
                mask_path = os.path.join(save_dir,self.get_frame_name(i_frame-1)+'.png')
            img_path = os.path.join(img_dir,self.get_frame_name(i_frame)+'.jpg')
            #print('deal with '+img_path)

            #open mask
            mask_img = Image.open(mask_path)
            if i_frame==1:
                palette=mask_img.getpalette()
            mask_img = np.array(mask_img)
            if i_frame==1:
                N_object = mask_img.max()

            #open image
            img = cv2.imread(img_path)
            g_height,g_width,_ = img.shape

            n_h,n_w,img,mask_img = self.resize_and_padding(img,mask_img,self.inputRes)

            """ cv2.imshow('ss',img)
            cv2.imshow('dd',mask_img)
            cv2.waitKey()
            exit() """

            if i_frame==1:
                print('frame num {} Img shape:{}-{},resize shape:{}-{}'.format(Nframes,g_height,g_width,n_h,n_w),img.shape)

            img = img.astype(np.float32)
            img = np.subtract(img, np.array(meanval, dtype=np.float32))
            height,width,_ = img.shape
            img = img.transpose((2,0,1))
            img = Variable(torch.from_numpy(img).float())

            output_mask = [None for _ in range(N_object+1)]

            for object_id in range(1,N_object+1):
                pre_mask = mask_img.copy()
                m1 = pre_mask==object_id
                m0 = pre_mask!=object_id
                pre_mask[m1]=1
                pre_mask[m0]=0

                pre_mask = pre_mask.astype(np.float32)
                pre_mask[pre_mask==1] = 100
                pre_mask[pre_mask==0] = -100
                pre_mask = pre_mask[np.newaxis,:,:]
                pre_mask =Variable(torch.from_numpy(pre_mask).float())

                input4channels = torch.cat([img,pre_mask],0)
                input4channels = input4channels.unsqueeze(0)

                if use_cuda:
                    input4channels = input4channels.cuda()
                
                outputs = net(input4channels)
                upsampler = torch.nn.Upsample(size=(height, width), mode='bilinear')
                outputs = upsampler(outputs)
                temp_bool = torch.le(outputs[0,0], outputs[0,1])
                outputs[0][1][temp_bool == 0] = self.large_neg
                output_mask[object_id] = outputs[0,1]

            flag = False
            for object_id in range(1, N_object+1):
                if flag == False:
                    max_mask = output_mask[object_id].clone()
                    flag = True
                else:
                    max_mask = torch.max(max_mask, output_mask[object_id])

            for object_id in range(1, N_object+1):

                output_mask[object_id].data[output_mask[object_id].data == self.large_neg] = 0
                output_mask[object_id].data[output_mask[object_id].data == max_mask.data] = 1
                output_mask[object_id] = output_mask[object_id].unsqueeze(0).unsqueeze(0).float()

            flag1=False

            for object_id in range(1, N_object+1):

                if flag1 == False:
                    final_output = output_mask[object_id].clone()
                    flag1=True

                final_output[output_mask[object_id]==1] = object_id

            #Get the palette and attach it
            final_label = np.uint8(final_output.data.cpu().numpy()[0][0])
            if n_h==-1:
                final_label = cv2.resize(final_label,(g_width,g_height),interpolation=cv2.INTER_NEAREST)
            else:
                final_ = final_label[:n_h,:n_w].copy()
                final_label = cv2.resize(final_,(g_width,g_height),interpolation=cv2.INTER_NEAREST)
            final_pil_image = Image.fromarray(final_label)
            final_pil_image.putpalette(palette)
            save_path = os.path.join(save_dir, self.get_frame_name(i_frame)+'.png')
            final_pil_image.save(save_path)
            print('-',end='',flush=True)
            #final_pil_image.show('ss')
            #input()
        print(' end\n')
   
           
    def video_test_without_train(self):
        #base_path = '/home/zjh/code/vseg/dataset/tianchiyusai/'
        #base_path = 'F:/vseg/dataset/media/tianchiyusai/tianchiyusai/'
        base_path = self.base_path
        set_type = 'test'
        #set_type = 'val'
        if set_type == 'test':
            set_file_path = os.path.join(base_path,'ImageSets','test.txt')
            result_dir = os.path.join(base_path,'result','test')
        else:
            set_file_path = os.path.join(base_path,'ImageSets','val.txt')
            result_dir = os.path.join('result','test')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        with open(set_file_path) as f:
            videos = [v.strip() for v in f.readlines()]

        #bulit net and load weight
        #model_path = 'pretrained/fparent_epoch-8.pth'   
        #model_path = 'offline_save_dir/lr_0.0005_wd_0.001/parent_epoch-1.pth'
        model_path = self.model_path
        state_dict = torch.load(model_path)    
        net = deeplab_resnet.Res_Deeplab_no_msc(2)
        net.load_state_dict(state_dict)
        if self.use_cuda:
            torch.cuda.set_device(0)
            net.cuda()
            print('use cuda')
        else:print('use cpu')
        net.eval()

        start_time=datetime.datetime.now()
        print('now we have {} videos to deal'.format(len(videos)))
        with torch.no_grad():
            for cnt,video in enumerate(videos):
                print('Predict in {} {}/{}'.format(video,cnt+1,len(videos)))
                first_mask_path = os.path.join(base_path,'Annotations',video,'00000.png')
                img_dir = os.path.join(base_path,'JPEGImages',video)
                save_dir = os.path.join(result_dir,video)

                shutil.copyfile(first_mask_path,os.path.join(save_dir,'00000.png'))
    
                self.test_without_train(net,first_mask_path,img_dir,save_dir=save_dir,use_cuda=self.use_cuda)
                present_time = datetime.datetime.now()
                remain = (present_time-start_time)*(len(videos)-cnt-1)/(cnt+1)
                print('{} remain {}'.format(present_time,remain))



if __name__ == '__main__':
    model_path = 'parent_epoch-6-19.pth'
    base_path = Path.db_offline_train_root_dir()
    online_test = Online_test(use_cuda=True,model_path=model_path,base_path=base_path)
    online_test.video_test_without_train()