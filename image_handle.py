import cv2
import numpy as np
from skimage import exposure
import os

aug_path_imgs = 'aug_path/imgs/'
aug_path_masks = 'aug_path/masks/'
raw_path = 'data/imgs/'
seg_path = 'data/masks/'
gamma_list = [0.6, 0.8, 1.2, 1.4, 1.6]
a_list = [135.0, 240.0, 320.0]
b_list = [5.0]

list_num = os.listdir(raw_path)
print(list_num)
k = 0
for i in range(len(list_num)):
    if list_num[i][-3:] == 'png' or list_num[i][-3:] == 'PNG' or list_num[i][-3:] == 'jpg' or list_num[i][-3:] == 'JPG':
        print('name:', list_num[i])
        img = cv2.imread(raw_path + list_num[i])
        seg_img = cv2.imread(seg_path + list_num[i], 0)
        print('img', np.array(img).shape, img.dtype)
        print('seg_mg', np.array(seg_img).shape, seg_img.dtype)

        #raw img
        cv2.imwrite(aug_path_imgs + str(k) + '.png', img)
        cv2.imwrite(aug_path_masks + str(k) + '.png', seg_img)
        k += 1

        #gamma transform
        for j in range(len(gamma_list)):
            new_imgs = exposure.adjust_gamma(img, gamma_list[j])
            cv2.imwrite(aug_path_imgs + str(k) + '.png', new_imgs)
            cv2.imwrite(aug_path_masks + str(k) + '.png', seg_img)
            k += 1

        #normalize
        for j in range(len(a_list)):
            for t in range(len(b_list)):
                new_imgs=cv2.normalize(img,dst=None,alpha=a_list[j],beta=b_list[t],norm_type=cv2.NORM_MINMAX)
                cv2.imwrite(aug_path_imgs + str(k) + '.png', new_imgs)
                cv2.imwrite(aug_path_masks + str(k) + '.png', seg_img)
                k += 1

        #equalize hist
        #img = cv2.imread(raw_path + list_num[i], cv2.IMREAD_GRAYSCALE)
        #new_imgs = cv2.equalizeHist(img)
        #cv2.imwrite(aug_path_imgs + str(k) + '.png', new_imgs)
        #cv2.imwrite(aug_path_masks + str(k) + '.png', seg_img)
        #new_imgs = cv2.imread(aug_path_masks + str(k) + '.png')
        #cv2.imwrite(aug_path_imgs + str(k) + '.png', new_imgs)

        #rotation
        #h_flip = cv2.flip(img, 1)
        #h_seg_img = cv2.flip(seg_img, 1)
        #print('h_flip size:', np.array(h_flip).shape)
        #cv2.imwrite(aug_path_imgs + str(k) + '.png', h_flip)
        #cv2.imwrite(aug_path_masks + str(k) + '.png', h_seg_img)
        #k += 1
        # Flipped Vertically
        #v_flip = cv2.flip(img, 0)
        #v_seg_img = cv2.flip(seg_img, 0)
        #cv2.imwrite(aug_path_imgs + str(k) + '.png', v_flip)
        #cv2.imwrite(aug_path_masks + str(k) + '.png', v_seg_img)
        #k += 1
        # Flipped Horizontally & Vertically
        #hv_flip = cv2.flip(img, -1)
        #hv_seg_img = cv2.flip(seg_img, -1)
        #cv2.imwrite(aug_path_imgs + str(k) + '.png', hv_flip)
        #cv2.imwrite(aug_path_masks + str(k) + '.png', hv_seg_img)
        #k += 1




















