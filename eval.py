import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn

from utils.dice_loss import dice_coeff
from utils.segLoss import SegmentationLosses

def eval_net(net, loader, model_name, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    #print('num_batch:', n_val)
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                if model_name == 'bisenet':
                    aux_pred0, aux_pred1, main_pred, smax_pred = net(imgs)
                elif model_name == 'danet':
                    main_pred = net(imgs)
                elif model_name == 'ddhnet_v2_b':
                    aux_1, aux_2, main_pred =  net(imgs)
                else:
                    mask_pred = net(imgs)
                softmax = nn.Softmax(dim=1)
                #masks_pred = torch.max(softmax(mask_pred), 1).indices

            if net.n_classes > 1:
                if model_name == 'bisenet':
                    tot += SegmentationLosses(cuda=True).CrossEntropyLoss(smax_pred, true_masks).item()
                elif model_name == 'danet':
                    tot += SegmentationLosses(cuda=True).CrossEntropyLoss(main_pred[1], true_masks).item()
                elif model_name == 'ddhnet_v2_b':
                    tot += SegmentationLosses(cuda=True).CrossEntropyLoss(main_pred, true_masks).item()
                else:
                    tot += SegmentationLosses(cuda=True).CrossEntropyLoss(mask_pred, true_masks).item()
            else:
                if model_name == 'bisenet':
                    smaxs_pred = torch.max(softmax(smax_pred), 1).indices
                    smaxs_pred = smaxs_pred.to(device=device, dtype=torch.float32)
                    smaxs_pred = (smaxs_pred > 0.5).float()
                    tot += dice_coeff(smaxs_pred, true_masks).item()
                elif model_name == 'ddhnet_v2_b':
                    main_preds = torch.max(softmax(main_pred), 1).indices
                    main_preds = main_preds.to(device=device, dtype=torch.float32)
                    main_preds = (main_preds > 0.5).float()
                    tot += dice_coeff(main_preds, true_masks).item()
                elif model_name == 'danet':
                    main_preds = torch.max(softmax(main_pred[1]), 1).indices
                    main_preds = main_preds.to(device=device, dtype=torch.float32)
                    main_preds = (main_preds > 0.5).float()
                    tot += dice_coeff(main_preds, true_masks).item()
                else:
                    masks_pred = torch.max(softmax(mask_pred), 1).indices
                #print(mask_pred.size())
                #print(true_masks.size())
                    masks_pred = masks_pred.to(device=device, dtype=torch.float32)

                    masks_pred = (masks_pred > 0.5).float()
                    tot += dice_coeff(masks_pred, true_masks).item()
            pbar.update()

    return tot / n_val
