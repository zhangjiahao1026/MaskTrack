"""
Author: Omkar Damle
Date: May 2018

Training code for Masktrack
 - using Deeplab Resnet-101
 - Offline training

"""
if __name__=='__main__':

    #from docopt import docopt
    import timeit
    import datetime
    from torch.utils.data import DataLoader
    import torch.backends.cudnn as cudnn
    import torch.optim as optim
    from torch.autograd import Variable

    from utility_functions import *
    from path import Path
    from dataloaders import davis17_offline_dataset as db17_offline
    import deeplab_resnet

    docstr = """
    Usage: 
        train.py [options]

    Options:
        -h, --help                  Print this message
        --NoLabels=<int>            The number of different labels in training data, Masktrack has 2 labels - foreground and background, including background [default: 2]
        --lr=<float>                Learning Rate [default: 0.001]
        --wtDecay=<float>          Weight decay during training [default: 0.001]
        --epochResume=<int>        Epoch from which to resume offline training [default: 0]
        --epochs=<int>             Training epochs [default: 1]
        --batchSize=<int>           Batch Size [default: 1]
    """

    #args = docopt(docstr, version='v0.1')

    ######################################################################################################################

    # Setting of parameters
    NoLabels = 2
    debug_mode = False
    use_cuda = False
    weight_decay = float(0.001)
    base_lr = float(0.0005)
    resume_epoch = int(0)  # Default is 0, change if want to resume
    nEpochs = int(15)  # Number of epochs for training (500.000/2079)
    batch_size = int(3)
    vbatch_size = 3
    db_root_dir = Path.db_offline_train_root_dir()
    nAveGrad = 4  # keep it even

    save_dir = os.path.join(Path.save_offline_root_dir(), 'lr_' + str(base_lr) + '_wd_' + str(weight_decay))

    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir))

    learnRate = base_lr

    ######################################################################################################################
    """Initialise the network"""
    net = deeplab_resnet.Res_Deeplab_no_msc(int(NoLabels))

    net.float()

    modelName = 'parent'
    if use_cuda:
        torch.cuda.set_device(0)

    if resume_epoch == 0:
        saved_state_dict = torch.load('pretrained/MS_DeepLab_resnet_pretrained_COCO_init.pth')
        """
        Last layer structure is modified according to the number of segmentation classes.
        For Mastrack, #classes=2
        """

        if int(NoLabels) != 21:
            for i in saved_state_dict:
                i_parts = i.split('.')
                if i_parts[1] == 'layer5':
                    saved_state_dict[i] = net.state_dict()[i]

                if i_parts[1] == 'conv1':
                    saved_state_dict[i] = torch.cat((saved_state_dict[i], torch.FloatTensor(64, 1, 7, 7).normal_(0,0.0001)), 1)

        net.load_state_dict(saved_state_dict)

        # Let us make it run on multiple GPUs!
        #if torch.cuda.device_count() > 1:
        #    print("Let's use", torch.cuda.device_count(), "GPUs!")
        #    net = nn.DataParallel(net)

    else:

        # Let us make it run on multiple GPUs!
        #if torch.cuda.device_count() > 1:
        #    print("Let's use", torch.cuda.device_count(), "GPUs!")
        #    net = nn.DataParallel(net)

        #print("Updating weights from: {}".format(
        #    os.path.join(save_dir, modelName + '_epoch-' + str(resume_epoch) + '.pth')))

        #net.load_state_dict(torch.load(os.path.join(save_dir, modelName + '_epoch-' + str(resume_epoch) + '.pth')))
        resume_path = 'pretrained/parent_epoch-6-19.pth'

        print("Updating weights from: {}".format(resume_path))
        net.load_state_dict(torch.load(resume_path))

    if use_cuda:
        print('use_cuda')
        net.cuda()
    else:
        print('CUDA not available')

    
    # net.train()

    optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(net), 'lr': base_lr},
                        {'params': get_10x_lr_params(net), 'lr': 10 * base_lr}],
                        lr=base_lr, momentum=0.9, weight_decay=weight_decay)

    if os.path.exists(os.path.join(save_dir, 'logs')) == False:
        os.mkdir(os.path.join(save_dir, 'logs'))


    ######################################################################################################################
    """Open file pointers for logging"""

    file_offline_loss = open(os.path.join(save_dir, 'logs/logs_offline_training_start_epoch_' + str(resume_epoch) + '.txt'), 'w+')
    #file_offline_train_precision = open(os.path.join(save_dir, 'logs/logs_offline_training_train_precision_start_epoch_' + str(resume_epoch) + '.txt'),'w+')
    #file_offline_train_recall = open(os.path.join(save_dir, 'logs/logs_offline_training_train_recall_start_epoch_' + str(resume_epoch) + '.txt'), 'w+')


    file_offline_val_loss = open(os.path.join(save_dir, 'logs/logs_offline_training_val_start_epoch_' + str(resume_epoch) + '.txt'), 'w+')
    #file_offline_val_precision = open(os.path.join(save_dir, 'logs/logs_offline_training_val_precision_start_epoch_' + str(resume_epoch) + '.txt'), 'w+')
    #file_offline_val_recall = open(os.path.join(save_dir, 'logs/logs_offline_training_val_recall_start_epoch_' + str(resume_epoch) + '.txt'), 'w+')

    loss_array = []
    loss_minibatch_array = []
    precision_train_array  = []
    recall_train_array = []

    loss_val_array = []
    precision_val_array = []
    recall_val_array = []

    aveGrad = 0



    ######################################################################################################################
    """Initialise the dataloaders"""

    dataset17_train = db17_offline.DAVIS17Offline(train=True, mini=False, mega=False, db_root_dir=Path.db_offline_train_root_dir(), transform=apply_custom_transform, inputRes=(480,854))
    dataloader17_train = DataLoader(dataset17_train, batch_size=batch_size, shuffle=True, num_workers=3)

    dataset17_val = db17_offline.DAVIS17Offline(train=False, mini=False, mega=False, db_root_dir=Path.db_offline_train_root_dir(), transform=apply_val_custom_transform,inputRes=(480,854))
    dataloader17_val = DataLoader(dataset17_val, batch_size=vbatch_size, shuffle=False, num_workers=3)

    lr_factor_array = [1,1,1,0.1,1,1,1,0.1,1,1,1,1,1,0.1,1,1,1,1]

    ######################################################################################################################
    print("Training Network")
    for epoch in range(resume_epoch+1, nEpochs+1):

        trainingDataSetSize = 0
        epochLoss = 0
        epochTrainIOU = 0
        #epochTrainPrecision = 0
        #epochTrainRecall = 0

        temp_Iou=0

        valDataSetSize = 0
        epochValLoss = 0
        epochValIOU = 0
        #epochValPrecision = 0
        #epochValRecall = 0

        start_time = timeit.default_timer()
        epoch_start_time = datetime.datetime.now()

        ######################################################################################################################
        total_train_batch = len(dataloader17_train)
        print('Training phase')
        print('len of loader: ' + str(total_train_batch))

        net.train()
        optimizer.zero_grad()
        aveGrad = 0
        for data_id, sample in enumerate(dataloader17_train):

            dic = net.state_dict()

            image = sample['image']
            anno = sample['gt']
            deformation = sample['deformation']
            
            # Making sure the mask input is similar to RGB values
            deformation[deformation==0] = -100
            deformation[deformation==1] = 100

            if debug_mode:
                cv2.imwrite('anno.png', anno[0].numpy().squeeze()*255)
                cv2.imwrite('defo.png', deformation[0].numpy().squeeze()*255)


            prev_frame_mask = Variable(deformation).float()
            inputs, gts = Variable(image), Variable(anno)

            if use_cuda:
                inputs, gts, prev_frame_mask = inputs.cuda(), gts.cuda(), prev_frame_mask.cuda()

            input_rgb_mask = torch.cat([inputs, prev_frame_mask], 1)
            noImages, noChannels, height, width = input_rgb_mask.shape
            


            output_mask = net(input_rgb_mask)
            
            upsampler = torch.nn.Upsample(size=(height, width), mode='bilinear')
            output_mask = upsampler(output_mask)
            
            if debug_mode:
                temp_out = np.zeros(output_mask[0][0].shape)
                temp_out[output_mask.data.cpu().numpy()[0][1] > output_mask.data.cpu().numpy()[0][0]] = 1
                cv2.imwrite('output.png',temp_out*255)

            loss1 = cross_entropy_loss(output_mask, gts)

            Iou_t = calculate_IOU(output_mask, gts)
            if data_id==0:
                temp_Iou = Iou_t
            else:
                temp_Iou = temp_Iou*0.999 + Iou_t*0.001
            epochTrainIOU += Iou_t
            now_time = datetime.datetime.now()
            remain_time = (now_time-epoch_start_time)*((total_train_batch-data_id-1)/(data_id+1))
            print('{} time remain {} epoch {} {}/{} train loss:{:.5f} Iou:{:.4f} lr:{} aveIou {:.4f}'.format(
                now_time,remain_time,epoch,(data_id+1)*batch_size,(total_train_batch)*batch_size,loss1.item(),Iou_t,learnRate,temp_Iou))
            #Precision_t = calculate_precision(output_mask, gts)
            #Recall_t = calculate_recall(output_mask, gts)
            #epochTrainPrecision += Precision_t
            #epochTrainRecall += Recall_t 
            #f1t = Precision_t*Recall_t
            #if f1t==0:
            #    F1_score=0
            #else:
            #    F1_score = (Precision_t*Recall_t)*2/(Precision_t+Recall_t)
            #print('loss:{:.5f} precision:{:.5f} recall:{:.5f} f1_score:{:.5f}'.format(loss1.item(),Precision_t,Recall_t,F1_score))

            loss_minibatch_array.append(loss1.item())

            epochLoss += loss1.item()
            trainingDataSetSize += 1

            #optimizer.zero_grad()
            #loss1.backward()
            #optimizer.step()

            # Backward the averaged gradient
            loss1 /= nAveGrad
            loss1.backward()
            aveGrad += 1

            # Update the weights once in nAveGrad forward passes
            if aveGrad % nAveGrad == 0:
                optimizer.step()
                optimizer.zero_grad()
                aveGrad = 0
            if (data_id+1)%10000==0:
                torch.save(net.state_dict(), os.path.join(save_dir, modelName + '_epoch-' + str(epoch) + '.pth'))
                print('stage saved')

        epochLoss = epochLoss / trainingDataSetSize
        #epochTrainPrecision = epochTrainPrecision / trainingDataSetSize
        #epochTrainRecall = epochTrainRecall / trainingDataSetSize
        epochTrainIOU = epochTrainIOU / trainingDataSetSize

        print('Epoch: ' + str(epoch) + ', Training Loss: ' + str(epochLoss) + '\n')
        #print('Epoch: ' + str(epoch) + ', Train Precision: ' + str(epochTrainPrecision) + '\n')
        #print('Epoch: ' + str(epoch) + ', Train Recall: ' + str(epochTrainRecall) + '\n')
        print('Epoch: ' + str(epoch) + ', Training IOU: ' + str(epochTrainIOU) + '\n')

        file_offline_loss.write(str(datetime.datetime.now())
                                +' Epoch: ' + str(epoch)
                                +', Loss: ' + str(epochLoss)
                                +', IOU: ' +str(epochTrainIOU)
                                +', lr: ' +str(learnRate)
                                + '\n')
        #file_offline_train_precision.write('Epoch: ' + str(epoch) + ', Precision: ' + str(epochTrainPrecision) + '\n')
        #file_offline_train_recall.write('Epoch: ' + str(epoch) + ', Recall: ' + str(epochTrainRecall) + '\n')

        loss_array.append(epochLoss)
        #precision_train_array.append(epochTrainPrecision)
        #recall_train_array.append(epochTrainRecall)

        file_offline_loss.flush()
        torch.save(net.state_dict(), os.path.join(save_dir, modelName + '_epoch-' + str(epoch) + '.pth'))


        ######################################################################################################################
        print('Validation phase')
        total_val_batch = len(dataloader17_val)
        aveGrad = 0
        net.eval()
        with torch.no_grad():
            for data_id, sample in enumerate(dataloader17_val):

                image = sample['image']
                anno = sample['gt']
                deformation = sample['deformation']

                deformation[deformation==0] = -100
                deformation[deformation==1] = 100

                prev_frame_mask = Variable(deformation, volatile=True).float()
                inputs, gts = Variable(image, volatile=True), Variable(anno, volatile=True)

                if use_cuda:
                    inputs, gts, prev_frame_mask = inputs.cuda(), gts.cuda(), prev_frame_mask.cuda()

                input_rgb_mask = torch.cat([inputs, prev_frame_mask], 1)

                noImages, noChannels, height, width = input_rgb_mask.shape
                
                output_mask = net(input_rgb_mask)

                upsampler = torch.nn.Upsample(size=(height, width), mode='bilinear')
                output_mask = upsampler(output_mask)
                

                loss1 = cross_entropy_loss(output_mask, gts)

                Iou_t = calculate_IOU(output_mask, gts)
                epochValIOU += Iou_t
                print('{} epoch {}  {}/{} val loss:{:5f} Iou:{:5f} lr:{}'.format(datetime.datetime.now(),epoch,(data_id+1)*vbatch_size,(total_val_batch)*vbatch_size,loss1.item(),Iou_t,learnRate))

                #epochValPrecision += calculate_precision(output_mask, gts)
                #epochValRecall += calculate_recall(output_mask, gts)
                epochValLoss += loss1.item()
                valDataSetSize += 1

        

        epochValLoss = epochValLoss / valDataSetSize
        #epochValPrecision = epochValPrecision / valDataSetSize
        #epochValRecall = epochValRecall / valDataSetSize
        epochValIOU = epochValIOU / valDataSetSize

        

        print('Epoch: ' + str(epoch) + ', Val Loss: ' + str(epochValLoss) + '\n')
        #print('Epoch: ' + str(epoch) + ', Val Precision: ' + str(epochValPrecision) + '\n')
        #print('Epoch: ' + str(epoch) + ', Val Recall: ' + str(epochValRecall) + '\n')
        print('Epoch: ' + str(epoch) + ', Val IOU: ' + str(epochValIOU) + '\n')

        
        file_offline_val_loss.write(str(datetime.datetime.now())
                                     +' Epoch: ' + str(epoch)
                                     + ', Loss: ' + str(epochValLoss) 
                                     +', IOU: ' +str(epochValIOU)
                                     +', lr: ' +str(learnRate)
                                     + '\n')
        #file_offline_val_precision.write('Epoch: ' + str(epoch) + ', Precision: ' + str(epochValPrecision) + '\n')
        #file_offline_val_recall.write('Epoch: ' + str(epoch) + ', Recall: ' + str(epochValRecall) + '\n')

        
        loss_val_array.append(epochValLoss)
        #precision_val_array.append(epochValPrecision)
        #recall_val_array.append(epochValRecall)

        
        file_offline_val_loss.flush()
        #file_offline_train_precision.flush()
        #file_offline_val_precision.flush()
        #file_offline_train_recall.flush()
        #file_offline_val_recall.flush()

        epochLoss = 0


        stop_time = timeit.default_timer()

        epoch_secs = stop_time - start_time
        epoch_mins = epoch_secs / 60
        epoch_hr = epoch_mins / 60

        print('This epoch took: ' + str(epoch_hr) + ' hours')

        
        #change_lr(optimizer, epoch)
        #read_lr(optimizer, save_dir)

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']*lr_factor_array[epoch-1]
        learnRate = learnRate*lr_factor_array[epoch-1]


        plot_loss1(loss_array, resume_epoch, epoch , save_dir)
        plot_loss1(loss_val_array, resume_epoch, epoch , save_dir, val=True)
        plot_loss_minibatch(loss_minibatch_array, save_dir)
        #plot_precision_recall(precision_train_array, recall_train_array, precision_val_array, recall_val_array, resume_epoch, epoch, save_dir)


    file_offline_loss.close()
    #file_offline_train_precision.close()
    #file_offline_train_recall.close()

    file_offline_val_loss.close()
    #file_offline_val_precision.close()
    #file_offline_val_recall.close()

    print('Offline training completed. Have a look at the plots to ensure hyperparameter selection is appropriate.')
    print('Need to fine-tune on each test video using online training.')