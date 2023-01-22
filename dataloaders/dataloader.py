import numpy as np
import matplotlib.pyplot as plt
import os
import random
import torch
from torch.nn import functional as F
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset, random_split
import cv2

import sys
sys.path.append('../utils')
from utils.utils import PILToTensor

import PIL
from PIL import Image
from PIL import ImageOps


PATH = 'C://Users//percy//Documents//FreiHand//dataset'
TRAIN_PATH = 'C://Users//percy//Documents//FreiHand//dataset//training//rgb2'
TRAIN_MASKS_PATH = 'C://Users//percy//Documents//FreiHand//dataset//training//mask2'
# TEST_PATH = 'C://Users//percy//Documents//FreiHand//dataset//evaluation//rgb'


def plot_mini_batch(imgs, masks, BATCH_SIZE=None):
    plt.figure(figsize=(20,10))

    for i in range(BATCH_SIZE):
        plt.subplot(4, 8, i+1)
        img=imgs[i,...].permute(1,2,0)

        print(masks.shape)
        mask=masks[i,...].permute(1,2,0)
        mask=mask[:,:,1]
        #print(mask.shape)
        plt.imshow(img)
        plt.axis('Off')

        plt.imshow(mask, alpha=0.5)

    plt.tight_layout()
    plt.show()


# Custom Dataset Class
class Hand_Dataset(Dataset):
    def __init__(self, data, masks=None, img_transforms=None, mask_transforms=None):
        '''
        data - train data path
        masks - train masks path
        '''

        self.train_data = data
        self.train_masks = masks

        self.img_transforms = img_transforms
        self.masks_transforms = mask_transforms

        self.images = sorted(os.listdir(self.train_data))
        self.masks = sorted(os.listdir(self.train_masks))
    
    
    def __len__(self):
        if self.train_masks is not None:
            assert len(self.images) == len(self.masks), 'not the same number of images and masks'
        
        return len(self.images)
    
    def __getitem__(self, idx):
        image_name = os.path.join(self.train_data, self.images[idx])
        img = Image.open(image_name)
        img = TF.equalize(img)

        trans = T.ToTensor()

        if self.img_transforms is not None:
            img = self.img_transforms(img)
        else:
            img = trans(img)


        if self.train_masks is not None:
            mask_name = os.path.join(self.train_masks, self.masks[idx])
            hand = Image.open(mask_name)
            hand = ImageOps.grayscale(hand)
            hand = torch.as_tensor(np.array(hand), dtype=torch.long)
            hand[hand < 100] = 0.0
            hand[hand >= 100] = 1.0
            H,W = hand.shape
            hand = torch.reshape(hand, (1,H,W))
            #background = torch.ones((1,H,W), dtype=torch.long) - hand
            #mask = torch.cat((background, hand), dim=0)
            mask = hand

            if self.masks_transforms is not None:
                mask = self.masks_transforms(mask)
            #print(mask.shape)
        
        return img, mask


transform_image = T.Compose([
    T.Resize([224,224]),
    T.ToTensor()
])

# transform_mask = T.Compose([
#     T.Resize([224,224]),
#     T.ToTensor()
# ])

if __name__ == "__main__":
    
    # Data Loader
    full_dataset = Hand_Dataset(TRAIN_PATH, 
                                TRAIN_MASKS_PATH, 
                                img_transforms = transform_image, 
                                mask_transforms = None)

    BATCH_SIZE = 32
    TRAIN_SIZE = int(len(full_dataset)*0.8)
    VAL_SIZE = len(full_dataset) - TRAIN_SIZE

    print("#Total batches during training --> TRAIN:",TRAIN_SIZE,"| VAL:",VAL_SIZE)

    train_dataset, val_dataset = random_split(full_dataset, [TRAIN_SIZE, VAL_SIZE])
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    imgs, masks = next(iter(train_loader))

    plot_mini_batch(imgs, masks, BATCH_SIZE=BATCH_SIZE)
