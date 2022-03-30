import argparse
import logging
import os
import cv2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.dataset import BasicDataset
from utils.segLoss import SegmentationLosses
from metric_eval import metric_eval
from models.segnet import SegNet
from models.aunet import AUNet_R16
from models.bisenet import BiSeNet
from models.nhbsnet import NHBSNet
from models.danet import DANet

model = './checkpoints/'
input_path = './data/test_imgs/'
output_path = './data/test_result/'
target_path = './data/test_masks/'
model_name = 'nhbsnet'
#SegNet: 'segnet'
#AUNet: 'aunet'
#BiSeNet: 'bisenet'
#DANet: 'danet'
#NHBSNet: 'nhbsnet'
loss = SegmentationLosses(cuda=True)
#loss_value = 0

def img_save_classes(img, img_name):
    #print('max_class:', np.max(img))
    for c in range(8):
        img_for_classes = np.zeros((img.shape))
        num = 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                    if img[i, j] == c:
                        num += 1
                        img_for_classes[i, j] = 255
        cv2.imwrite(str(c)+ img_name+'.png', img_for_classes)

def mask_to_image(mask):
    return (mask * 30).astype(np.uint8)

def predict_img(net,
                full_img,
                device,
                target_file = './aug_path/mask/402.png'):
    net.eval()
    #print('full_img:', np.array(full_img), np.max(full_img), np.min(full_img))
    #print(np.array(full_img))
    img = torch.from_numpy(BasicDataset.preprocess(full_img, 1))
    #print(target_file)
    target = Image.open(target_file)
    #print(np.array(full_img))
    #print('target_max:', np.max(np.array(target)))
    target = torch.from_numpy(BasicDataset.preprocess(target, 1))
    t = target[0].long()
    one_hot = F.one_hot(t, num_classes=8).permute(2, 0, 1).unsqueeze(0)
    target_img = np.array(target)
    img_save_classes(target_img[0], 'true')
    #criterion = SegmentationLosses(cuda=True)

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    #print('img_size:', img.size())

    with torch.no_grad():
        if model_name == 9:
            aux_pred0, aux_pred1, main_pred, output = net(img)
            #print(output.size(), target.int().cuda().size())    
        elif model_name == 10:
            main_pred = net(img)
            output = main_pred[1]
        else:
            output = net(img)

        CEloss = loss.CrossEntropyLoss(output, target.int().cuda())
        #ssCEloss = loss.CrossEntropyLoss(output, target.int().cuda())
        #print(CEloss)
        loss_value = CEloss.item()
        #print('output:', output, output.size())

        if net.n_classes > 1:
            softmax = nn.Softmax(dim=1)

            #print('masks_pred:', torch.max(softmax(output), 1).indices)
            probs = torch.max(softmax(output), 1).indices[0]
            #print(probs.size())
            #probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)
            #print('sigmoid:', probs)
            #softmax = nn.Softmax(dim=0)
            #print(output.size())

        probs = probs.squeeze(0)

        full_mask = probs.cpu().numpy()
        img_save_classes(full_mask, 'pred')

        voe = []
        rvd = []
        precision = []
        recall = []
        dice = []
        h_dice = []
        haurdorff = []
        ave_haurdorff = []

        for c in range(8):
            #print('class for'+str(c))
            VOE, RVD, Precision, Recall, Dice, H_Dice, Haurdorff, Ave_Haurdorff = metric_eval(str(c)+'pred.png', str(c)+'true.png')
            #print(DSI, VOE, RVD, Precision, Recall)
            voe.append(VOE)
            rvd.append(RVD)
            precision.append(Precision)
            recall.append(Recall)
            dice.append(Dice)
            h_dice.append(H_Dice)
            haurdorff.append(Haurdorff)
            ave_haurdorff.append(Ave_Haurdorff)

        s =  Image.fromarray((full_mask * 255).astype(np.float))
        #cv2.imwrite('249.png', (full_mask * 20).astype(np.float))
    return full_mask, voe, rvd, precision, recall, dice, h_dice, haurdorff, ave_haurdorff, loss_value

def predict(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cpu':
        cuda_device = False
    else:
        cuda_device = True
    if model_name == 'segnet':
        net = SegNet(n_channels=3, n_classes=8)
    elif model_name == 'aunet':
        net = AUNet_R16(n_channels=3, n_classes=8, learned_bilinear=True)
    elif model_name == 'bisenet':
        net = BiSeNet(n_classes=8, n_channels=3)
    elif model_name == 'danet':
        net = DANet(n_classes=8, n_channels=3, backbone='resnet50')
    elif model_name == 'nhbsnet':
        net = NHBSNet(n_classes=8, n_channels=3, cuda_device=cuda_device)
        
    logging.info("Loading model {}".format(model_path))
    #device = 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(model_path, map_location=device))

    logging.info("Model loaded !")

    in_files = os.listdir(input_path)

    total_voe = np.zeros((len(in_files), 8))
    total_rvd = np.zeros((len(in_files), 8))
    total_precision = np.zeros((len(in_files), 8))
    total_recall = np.zeros((len(in_files), 8))
    total_dice = np.zeros((len(in_files), 8))
    total_h_dice = np.zeros((len(in_files), 8))
    total_haurdorff = np.zeros((len(in_files), 8))
    total_ave_haurdorff = np.zeros((len(in_files), 8))
    Loss = 0

    for i in range(len(in_files)):
        print(i, input_path + in_files[i])

        img = Image.open(input_path + in_files[i])
        mask, voe, rvd, precision, recall, dice, h_dice, haurdorff, ave_haurdorff, loss_value = predict_img(net=net,
                           full_img=img,
                           device=device,
                           target_file = target_path + in_files[i])
        Loss += loss_value

        mask_img = mask_to_image(mask)
        print(i, 'dice', dice)
        print(i, 'haurdorff', haurdorff)
        for j in range(8):
            total_voe[i][j] = voe[j]
            total_rvd[i][j] = rvd[j]
            total_precision[i][j] = precision[j]
            total_recall[i][j] = recall[j]
            total_dice[i][j] = dice[j]
            total_h_dice[i][j] = h_dice[j]
            total_haurdorff[i][j] = haurdorff[j]
            total_ave_haurdorff[i][j] = ave_haurdorff[j]

        cv2.imwrite(output_path + in_files[i], mask_img) 

    Loss /= len(in_files)
    print('Loss:', Loss)
    print('voe-mean:', np.mean(total_voe[:, 0]), np.mean(total_voe[:, 1]), np.mean(total_voe[:, 2]), np.mean(total_voe[:, 3]), np.mean(total_voe[:, 4]), np.mean(total_voe[:, 5]),np.mean(total_voe[:, 6]),np.mean(total_voe[:, 7]))
    print('voe-std:', np.std(total_voe[:, 0]), np.std(total_voe[:, 1]), np.std(total_voe[:, 2]), np.std(total_voe[:, 3]), np.std(total_voe[:, 4]), np.std(total_voe[:, 5]),np.std(total_voe[:, 6]),np.std(total_voe[:, 7]))
    print('rvd-mean:', np.mean(total_rvd[:, 0]), np.mean(total_rvd[:, 1]), np.mean(total_rvd[:, 2]), np.mean(total_voe[:, 3]), np.mean(total_rvd[:, 4]), np.mean(total_rvd[:, 5]),np.mean(total_rvd[:, 6]),np.mean(total_rvd[:, 7]))
    print('rvd-std:', np.std(total_rvd[:, 0]), np.std(total_rvd[:, 1]), np.std(total_rvd[:, 2]), np.std(total_voe[:, 3]), np.std(total_rvd[:, 4]), np.std(total_rvd[:, 5]),np.std(total_rvd[:, 6]),np.std(total_rvd[:, 7]))
    print('precision-mean:', np.mean(total_precision[:, 0]), np.mean(total_precision[:, 1]), np.mean(total_precision[:, 2]), np.mean(total_precision[:, 3]), np.mean(total_precision[:, 4]), np.mean(total_precision[:, 5]),np.mean(total_precision[:, 6]),np.mean(total_precision[:, 7]))
    print('precision-std:', np.std(total_precision[:, 0]), np.std(total_precision[:, 1]), np.std(total_precision[:, 2]), np.std(total_precision[:, 3]), np.std(total_precision[:, 4]), np.std(total_precision[:, 5]),np.std(total_precision[:, 6]),np.std(total_precision[:, 7]))
    print('recall-mean:', np.mean(total_recall[:, 0]), np.mean(total_recall[:, 1]), np.mean(total_recall[:, 2]), np.mean(total_recall[:, 3]), np.mean(total_recall[:, 4]), np.mean(total_recall[:, 5]),np.mean(total_recall[:, 6]),np.mean(total_recall[:, 7]))
    print('recall-std:', np.std(total_recall[:, 0]), np.std(total_recall[:, 1]), np.std(total_recall[:, 2]), np.std(total_recall[:, 3]), np.std(total_recall[:, 4]), np.std(total_recall[:, 5]),np.std(total_recall[:, 6]),np.std(total_recall[:, 7]))
    print('dice-mean:', np.mean(total_dice[:, 0]), np.mean(total_dice[:, 1]), np.mean(total_dice[:, 2]), np.mean(total_dice[:, 3]), np.mean(total_dice[:, 4]), np.mean(total_dice[:, 5]),np.mean(total_dice[:, 6]),np.mean(total_dice[:, 7]))
    print('dice-std:', np.std(total_dice[:, 0]), np.std(total_dice[:, 1]), np.std(total_dice[:, 2]), np.std(total_dice[:, 3]), np.std(total_dice[:, 4]), np.std(total_dice[:, 5]),np.std(total_dice[:, 6]),np.std(total_dice[:, 7]))
    print('haurdorff-mean:', np.mean(total_haurdorff[:, 0]), np.mean(total_haurdorff[:, 1]), np.mean(total_haurdorff[:, 2]), np.mean(total_haurdorff[:, 3]), np.mean(total_haurdorff[:, 4]), np.mean(total_haurdorff[:, 5]),np.mean(total_haurdorff[:, 6]),np.mean(total_haurdorff[:, 7]))
    print('haurdorff-std:', np.std(total_haurdorff[:, 0]), np.std(total_haurdorff[:, 1]), np.std(total_haurdorff[:, 2]), np.std(total_haurdorff[:, 3]), np.std(total_haurdorff[:, 4]), np.std(total_haurdorff[:, 5]),np.std(total_haurdorff[:, 6]),np.std(total_haurdorff[:, 7]))
    print('ave_haurdorff-mean:', np.mean(total_ave_haurdorff[:, 0]), np.mean(total_ave_haurdorff[:, 1]), np.mean(total_ave_haurdorff[:, 2]), np.mean(total_ave_haurdorff[:, 3]), np.mean(total_ave_haurdorff[:, 4]), np.mean(total_ave_haurdorff[:, 5]),np.mean(total_ave_haurdorff[:, 6]),np.mean(total_ave_haurdorff[:, 7]))
    print('ave_haurdorff-std:', np.std(total_ave_haurdorff[:, 0]), np.std(total_ave_haurdorff[:, 1]), np.std(total_ave_haurdorff[:, 2]), np.std(total_ave_haurdorff[:, 3]), np.std(total_ave_haurdorff[:, 4]), np.std(total_ave_haurdorff[:, 5]),np.std(total_ave_haurdorff[:, 6]),np.std(total_ave_haurdorff[:, 7]))
    return

if __name__ == "__main__":
    for i in range(1, 2):
        model_path = model+'CP_epoch'+str(i)+'.pth'
        print(model_path)
        predict(model_path)