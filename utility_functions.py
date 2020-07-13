"""
Author: Omkar Damle
Date: May 2018

All the functions required for offline and online training of Deeplab Resnet for MaskTrack method
"""

import numpy as np
import scipy.stats as scipystats
import torch.nn as nn
import torch

import os
import matplotlib.pyplot as plt
#import scipy.misc as sm
import cv2
import random


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batch  layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = []

    b.append(model.Scale.conv1)
    b.append(model.Scale.bn1)
    b.append(model.Scale.layer1)
    b.append(model.Scale.layer2)
    b.append(model.Scale.layer3)
    b.append(model.Scale.layer4)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj += 1
                if k.requires_grad:
                    yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """

    b = []
    b.append(model.Scale.layer5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i

def calculate_IOU(output_mask, gts):
    with torch.no_grad():
        Iou = 0

        no_images = len(output_mask)

        for image_id in range(no_images):
            eps = 0.00001
            output_label = (output_mask[image_id][1]>=output_mask[image_id][0]).float()
            gt = gts[image_id].squeeze(0)
            inter = torch.dot(output_label.view(-1), gt.view(-1))+ eps
            union = torch.sum(output_label) + torch.sum(inter) - inter + eps
            temp_Iou = inter.float()/ union.float()
            temp_Iou = temp_Iou.item()
            Iou+=temp_Iou
    return Iou/no_images
        

def calculate_precision(output_mask, gts, save_imgs=0):

    precision = 0

    no_images = len(output_mask)

    for image_id in range(no_images):

        output_label = torch.zeros(output_mask[0][0].data.cpu().numpy().shape)

        output_label[output_mask[image_id][1].data.cpu() >= output_mask[image_id][0].data.cpu()] = 1
        output_label[output_mask[image_id][1].data.cpu() < output_mask[image_id][0].data.cpu()] = 0

        gt = gts[image_id].squeeze(0)

        h, w = output_label.shape
        tmp = output_label.clone().cpu().numpy()
        output_pixels = np.sum(tmp)

        tmp[gt.data.cpu().numpy() == np.zeros((h,w))] = 0
        correct_mask_pixels = np.sum(tmp)


        if output_pixels == 0:
            temp_precision=0
        else:
            temp_precision = float(correct_mask_pixels) / output_pixels

        precision += temp_precision

    return float(precision)/no_images


def calculate_recall(output_mask, gts, save_imgs=0):

    recall = 0

    no_images = len(output_mask)

    for image_id in range(no_images):

        output_label = torch.zeros(output_mask[0][0].data.cpu().numpy().shape)

        output_label[output_mask[image_id][1].data.cpu() >= output_mask[image_id][0].data.cpu()] = 1
        output_label[output_mask[image_id][1].data.cpu() < output_mask[image_id][0].data.cpu()] = 0

        gt = gts[image_id].squeeze(0)

        h, w = output_label.shape
        tmp = output_label.clone().cpu().numpy()
        gt_tmp = gt.data.cpu().numpy()
        gt_pixels = np.sum(gt_tmp)

        tmp[gt_tmp == np.zeros((h,w))] = 0
        correct_mask_pixels = np.sum(tmp)

        if gt_pixels==0:
            temp_recall = 1
        else:
            temp_recall = float(correct_mask_pixels) / gt_pixels

        recall += temp_recall

    return float(recall)/no_images


def cross_entropy_loss(output, labels):
    """According to Pytorch documentation, nn.CrossEntropyLoss combines nn.LogSoftmax and nn.NLLLoss
    For loss,
        first argument should be class scores with shape: N,C,h,w
        second argument should be class labels with shape: N,h,w
    Assumes labels are binary
    """


    ce_loss = nn.CrossEntropyLoss()
    images, channels, height, width = output.data.shape
    loss = ce_loss(output, labels.long().view(images, height, width))
    return loss

def cross_entropy_loss_weighted(output, labels):

    temp = labels.data.cpu().numpy()
    freqCount = scipystats.itemfreq(temp)
    total = freqCount[0][1]+freqCount[1][1]
    perc_1 = freqCount[1][1]/total
    perc_0 = freqCount[0][1]/total

    weight_array = [perc_1, perc_0]

    if torch.cuda.is_available():
        weight_tensor = torch.FloatTensor(weight_array).cuda()
    else:
        weight_tensor = torch.FloatTensor(weight_array)

    ce_loss = nn.CrossEntropyLoss(weight=weight_tensor)
    images, channels, height, width = output.data.shape
    loss = ce_loss(output, labels.long().view(images, height, width))
    return loss

def plot_loss(loss_array, seq_name, nEpochs, save_dir):
    objs = len(loss_array)-1

    if os.path.exists(os.path.join(save_dir,'plots')) == False:
        os.mkdir(os.path.join(save_dir,'plots'))

    for obj_id in range(1,objs+1):
        x_axis = range(1, nEpochs+1)
        y_axis = loss_array[obj_id]
        plt.xlabel('Number of Epochs')
        plt.ylabel('Epoch loss')
        plt.plot(x_axis, y_axis)
        plt.savefig(os.path.join(save_dir, 'plots', seq_name + str(obj_id) + '.png'))
        plt.clf()

def plot_loss1(loss_array, resume_epoch, nEpochs, save_dir, val=False, online=False,seq_name = None, object_id=-1):

    if online:
        if os.path.exists(os.path.join(save_dir,'plots', seq_name, str(object_id))) == False:
            os.makedirs(os.path.join(save_dir,'plots',seq_name, str(object_id)))
    else:
        if os.path.exists(os.path.join(save_dir,'plots')) == False:
            os.mkdir(os.path.join(save_dir,'plots'))

    x_axis = range(resume_epoch + 1, nEpochs+1)
    y_axis = loss_array
    plt.xlabel('Number of Epochs')

    if val:
        plt.ylabel('Val Epoch loss')
        plt.plot(x_axis, y_axis)
        plt.savefig(os.path.join(save_dir, 'plots', 'val_loss_plot.png'))
    else:
        plt.ylabel('Train Epoch loss')
        plt.plot(x_axis, y_axis)
        if online:
            plt.savefig(os.path.join(save_dir, 'plots', seq_name, str(object_id), 'loss_plot.png'))
        else:
            plt.savefig(os.path.join(save_dir, 'plots', 'loss_plot.png'))

    plt.clf()

def plot_loss_minibatch(loss_array, save_dir, online=False, seq_name = None, object_id = -1):
    length = len(loss_array)
    plt.xlabel('minibatch number')
    plt.ylabel('loss')
    plt.plot(range(length),loss_array)

    if online:
        plt.savefig(os.path.join(save_dir, 'plots', seq_name, str(object_id), 'loss_minibatch_plot.png'))
    else:
        plt.savefig(os.path.join(save_dir, 'plots', 'loss_minibatch_plot.png'))

    plt.clf()

def plot_precision_recall(train_precision, train_recall, val_precision=None, val_recall=None, resume_epoch = 0, nepochs = -1, save_dir=None, online=False, seq_name = None, object_id = -1):
    assert len(range(resume_epoch + 1, nepochs+1)) == len(train_precision)
    xaxis = range(resume_epoch + 1, nepochs+1)
    plt.plot(xaxis, train_precision, label = "train_precision")
    plt.plot(xaxis, train_recall, label = "train_recall")

    if not online:
        plt.plot(xaxis, val_precision, label = "val_precision")
        plt.plot(xaxis, val_recall, label = "val_recall")

    plt.legend()

    if online:
        plt.savefig(os.path.join(save_dir, 'plots', seq_name, str(object_id),'accuracies.png'))
    else:
        plt.savefig(os.path.join(save_dir, 'plots', 'accuracies.png'))

    plt.clf()

def change_lr(optimizer, epoch):

    if epoch%2==0:
        return

    factor = 1

    print('Decreasing LR by: ' + str(factor))
    for param_group in optimizer.param_groups:
        #print(param_group['lr'])
        param_group['lr'] = param_group['lr']*factor
    #print('Done changing LR')

def read_lr(optimizer, save_dir):
    file = open(os.path.join(save_dir, 'lr_factor.txt'))
    lr_factor = float(file.readline())
    print(lr_factor)
    #asd

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']*lr_factor


def apply_transform_image(image, rot,sc, horz_flip, inputRes=None):

    meanval = (104.00699, 116.66877, 122.67892)

    if inputRes is not None:
        image = cv2.resize(image, inputRes)

    h, w = image.shape[:2]

    center = (w / 2, h / 2)
    assert (center != 0)  # Strange behaviour warpAffine
    M = cv2.getRotationMatrix2D(center, rot, sc)

    image = np.array(image, dtype=np.float32)
    image = np.subtract(image, np.array(meanval, dtype=np.float32))

    flagval = cv2.INTER_CUBIC
    image = cv2.warpAffine(image, M, (w,h),flags=flagval)

    if horz_flip:
        image = cv2.flip(image,flipCode=1)

    if image.ndim == 2:
        image=image[:,:,np.newaxis]

    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W

    image = image.transpose((2,0,1))
    image = torch.from_numpy(image)

    return image


def apply_transform_anno(annotation, rot,sc, horz_flip, inputRes=None):

    if inputRes is not None:
        annotation = cv2.resize(annotation, inputRes,interpolation=cv2.INTER_NEAREST)
    h, w = annotation.shape[:2]
    center = (w / 2, h / 2)
    assert (center != 0)  # Strange behaviour warpAffine
    M = cv2.getRotationMatrix2D(center, rot, sc)

    annotation = np.array(annotation, dtype=np.float32)

    flagval = cv2.INTER_NEAREST
    annotation = cv2.warpAffine(annotation, M, (w,h), flags=flagval)

    if horz_flip:
        annotation = cv2.flip(annotation, flipCode=1)

    if annotation.ndim == 2:
        annotation=annotation[:,:,np.newaxis]

    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    annotation = annotation.transpose((2,0,1))
    annotation = torch.from_numpy(annotation)


    dividing_factor = annotation.max()

    if dividing_factor == 0:
        dividing_factor = 1

    annotation = annotation/dividing_factor

    return annotation

def resize_and_padding(img,label,df,inputRes=None):
    if inputRes==None:
        return img,label,df
    w_div_h = float(inputRes[1])/inputRes[0]
    im_w_h = float(label.shape[1])/label.shape[0]
    if abs(w_div_h-im_w_h)<0.1:
        img = cv2.resize(img,(inputRes[1],inputRes[0]),interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label,(inputRes[1],inputRes[0]),interpolation=cv2.INTER_NEAREST)
        df = cv2.resize(df,(inputRes[1],inputRes[0]),interpolation=cv2.INTER_NEAREST)
        return img,label,df
    n_h=0
    n_w=0
    if im_w_h > w_div_h:
        scale = inputRes[1]/label.shape[1]
        conv_h = int(label.shape[0]*scale)
        n_h = conv_h
        n_w = inputRes[1]
    else:
        scale = inputRes[0]/label.shape[0]
        conv_w = int(label.shape[1]*scale)
        n_h = inputRes[0]
        n_w = conv_w
    img_s = cv2.resize(img,(n_w,n_h),interpolation=cv2.INTER_CUBIC)
    label_s = cv2.resize(label,(n_w,n_h),interpolation=cv2.INTER_NEAREST)
    df_s = cv2.resize(df,(n_w,n_h),interpolation=cv2.INTER_NEAREST)

    img = np.zeros((inputRes[0],inputRes[1],3),dtype=np.uint8)
    label = np.zeros((inputRes[0],inputRes[1]),dtype=np.uint8)
    df = np.zeros((inputRes[0],inputRes[1]),dtype=np.uint8)

    img[:n_h,:n_w,:] = img_s
    label[:n_h,:n_w] = label_s
    df[:n_h,:n_w] = df_s
    return img,label,df


def apply_custom_transform(img, label, df, inputRes=None):

    rots = (-30, 30)
    scales = (0.5, 1.3)

    """Data augmentations"""
    rot = (rots[1] - rots[0]) * random.random() - (rots[1] - rots[0]) / 2
    sc = (scales[1] - scales[0]) * random.random() + scales[0]

    horz_flip = False
    if random.random() < 0.5:
        horz_flip = True
    #resize and padding 
    img,label,df = resize_and_padding(img,label,df,inputRes)

    img=apply_transform_image(img, rot, sc, horz_flip, inputRes=None)
    label = apply_transform_anno(label, rot, sc, horz_flip, inputRes=None)
    df = apply_transform_anno(df, rot, sc, horz_flip, inputRes=None)

    return img,label,df

def apply_val_transform_image(image,inputRes=None):
    meanval = (104.00699, 116.66877, 122.67892)

    if inputRes is not None:
        image = cv2.resize(image, inputRes)

    image = np.array(image, dtype=np.float32)
    image = np.subtract(image, np.array(meanval, dtype=np.float32))



    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W

    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image)

    return image

def apply_val_transform_anno(annotation, inputRes=None):

    #print(annotation)

    if inputRes is not None:
        annotation = cv2.resize(annotation, inputRes,interpolation=cv2.INTER_NEAREST)

    annotation = np.array(annotation, dtype=np.float32)

    if annotation.ndim == 2:
        annotation=annotation[:,:,np.newaxis]

    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    annotation = annotation.transpose((2,0,1))
    annotation = torch.from_numpy(annotation)

    dividing_factor = annotation.max()

    if dividing_factor == 0:
        dividing_factor = 1

    annotation = annotation/dividing_factor

    return annotation

def apply_val_custom_transform(img, label, df, inputRes=None):
    #resize and padding 
    img,label,df = resize_and_padding(img,label,df,inputRes)

    img = apply_val_transform_image(img,inputRes=None)
    label = apply_val_transform_anno(label,inputRes=None)
    df = apply_val_transform_anno(df,inputRes=None)

    return img, label, df