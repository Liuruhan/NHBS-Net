import argparse
import logging
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from eval import eval_net
from utils.dice_loss import dice_coeff
from utils.facal_loss import Facal_loss
from utils.dataset import BasicDataset
from utils.segLoss import SegmentationLosses

from models.segnet import SegNet
from models.aunet import AUNet_R16
from models.bisenet import BiSeNet
from models.nhbsnet import NHBSNet
from models.danet import DANet

tr_dir_img = 'aug_path/imgs/'
tr_dir_mask = 'aug_path/masks/'
te_dir_img = 'data/test_imgs/'
te_dir_mask = 'data/test_masks/'
dir_checkpoint = 'checkpoints/'
model_name = 'nhbsnet'
#SegNet: 'segnet'
#AUNet: 'aunet'
#BiSeNet: 'bisenet'
#DANet: 'danet'
#NHBSNet: 'nhbsnet'
print(dir_checkpoint)

def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.0001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5,
              alpha=0.5):

    train = BasicDataset(tr_dir_img, tr_dir_mask, img_scale)
    val = BasicDataset(te_dir_img, te_dir_mask, img_scale)
    n_val = len(val)
    n_train = len(train)
    print(n_train, n_val)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    writer = SummaryWriter(comment=f'50_Unet_LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Images scaling:  {img_scale}
    ''')
    #Device:          {device.type}
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-12, momentum=0.95)
    #optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-12)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)

    if net.n_classes > 1:
        #criterion = nn.CrossEntropyLoss()
        criterion = SegmentationLosses(cuda=True)

    else:
        criterion = nn.BCEWithLogitsLoss()
        #criterion = nn.BCELoss()
        #criterion = dice_coeff()
    times = 0
    for epoch in range(epochs):
        net.train()
        epoch_mask = True
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float64 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                #print('img_shape:', imgs.shape)
                if model_name == 'bisenet':
                    aux_pred0, aux_pred1, main_pred, smax_pred = net(imgs)
                    aux_loss0 = criterion.CrossEntropyLoss(aux_pred0, true_masks)+criterion.FocalLoss(aux_pred0, true_masks, gamma=2, alpha=0.5)
                    aux_loss1 = criterion.CrossEntropyLoss(aux_pred1, true_masks)+criterion.FocalLoss(aux_pred1, true_masks, gamma=2, alpha=0.5)
                    main_loss = criterion.CrossEntropyLoss(main_pred, true_masks)+criterion.FocalLoss(main_pred, true_masks, gamma=2, alpha=0.5)
                    loss = (aux_loss0 + aux_loss1 + main_loss)/3
                elif model_name == 'danet':
                    main_pred = net(imgs)
                    aux_loss = criterion.CrossEntropyLoss(main_pred[0], true_masks)+criterion.FocalLoss(main_pred[0], true_masks, gamma=2, alpha=0.5)
                    main_loss = criterion.CrossEntropyLoss(main_pred[1], true_masks)+criterion.FocalLoss(main_pred[1], true_masks, gamma=2, alpha=0.5)
                    loss = (aux_loss + main_loss)/2
                else:
                    masks_pred = net(imgs)
                    loss = criterion.CrossEntropyLoss(masks_pred, true_masks) + criterion.FocalLoss(masks_pred, true_masks, gamma=2, alpha=0.5)

                #print(loss)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

                if global_step % ((len(train)+len(val)) // (10 *batch_size)) == 0:
                    #print(global_step)
                    softmax = nn.Softmax(dim=1)
                    #torch.max(softmax(x), 1).indices
                    if  epoch_mask == True:
                        #print(global_step)
                        print('Images save!')
                        if model_name == 'bisenet':
                            pred_masks = torch.max(softmax(smax_pred), 1).indices
                        elif model_name == 'danet':
                            pred_masks = torch.max(softmax(main_pred[1]), 1).indices
                        else:
                            pred_masks = torch.max(softmax(masks_pred), 1).indices
                        #print('masks_pred:', torch.max(softmax(masks_pred), 1).indices)
                        #print('true_mask:', true_masks)#
                        pred_mask = pred_masks.cpu().numpy()
                        true_mask = true_masks.cpu().numpy()
                        for t in range(0):
                            pred = pred_mask[t] * 20
                            true = true_mask[t] * 20
                            cv2.imwrite(str(epoch)+str(times)+'pred.png', pred)
                            cv2.imwrite(str(epoch)+str(times)+'true.png', true)
                        epoch_mask = False

                    print('loss:', loss.item())
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        #writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                    val_score = eval_net(net, val_loader, model_name, device)
                    print('Validation Dice Coeff:', val_score)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            val_score = eval_net(net, val_loader, model_name, device)
            print('Validation Dice Coeff:', val_score)
            logging.info(f'Checkpoint {epoch + 1} saved !')
            print('Validation Dice Coeff:', val_score)

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=11,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=4,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cpu':
        cuda_device = False
    else:
        cuda_device = True
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
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

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n')
                 #f'\t{"Bilinear" if net.bilinear else "Dilated conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)