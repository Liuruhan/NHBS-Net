from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        #print('id:', self.ids)
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        #print(pil_img.size)
        img_nd = np.array(pil_img)
        #print(img_nd.shape)

        if len(img_nd.shape) == 2 :
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        if img_trans.max() == 255:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        #print('index:', idx)
        mask_file = glob(self.masks_dir + idx + '.png')
        img_file = glob(self.imgs_dir + idx + '.png')
        #print('mask_file:', mask_file)
        #print('img_file:', img_file)
        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)
        #print('img:', img.shape)
        #print('mask:', mask[0].shape)
        #print('mask_M:', np.max(mask[0]))
        #print('img:', torch.from_numpy(img).type(torch.ByteTensor), 'mask:', torch.from_numpy(mask[0]).type(torch.ByteTensor))
        return {'image': torch.from_numpy(img).type(torch.ByteTensor), 'mask': torch.from_numpy(mask[0]).type(torch.ByteTensor)}



