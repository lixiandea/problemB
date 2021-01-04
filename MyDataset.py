
import os
import random

import numpy as np
import torch
from skimage.io import imread
from torch.utils.data import Dataset
from torch.utils.data import random_split
import cv2  
from torch.utils.data import DataLoader
from tqdm import tqdm


class MyDataset(Dataset):
    """Brain MRI dataset for FLAIR abnormality segmentation"""
    
    in_channels = 3
    out_channels = 1

    def __init__(
        self,
        images_dir,
        image_size=256,
        seed=42,
    ):
        # read images
        self.pics = []
        self.masks = []
        pic_dir = os.path.join(images_dir, "src")
        label_dir = os.path.join(images_dir, "label")
        print("------{} pics------".format(len(os.listdir(pic_dir))))
        for i in tqdm(range(len(os.listdir(pic_dir)))):
        # for i in tqdm(range(10)):
            self.pics.append( os.path.join(pic_dir, str(i) + '.png'))
            self.masks.append(os.path.join(label_dir, str(i) + '.png'))

    def __len__(self):
        return len(self.pics)

    def __getitem__(self, idx):
        pic =  cv2.imread(self.pics[idx]).transpose(2,0,1).astype(np.float32)
        mask = cv2.imread(self.masks[idx], 2)[np.newaxis, :, :].astype(np.float32)

        return torch.from_numpy(pic), torch.from_numpy(mask)

    # def normalization(self, img):
    #     for i in range(3):
    #         img[:,:,i] = (img[:,:,i] - self.mu[i])/self.sigma[i]
    #     return img
if __name__ == "__main__":
    dataset = MyDataset("dataSet/train")
    batchSize = 16
    validation_split = 2
    shuffle_dataset = True
    random_seed = 42
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=4, batch_size=128)
    val_loader = DataLoader(val_dataset, shuffle=False, num_workers=1, batch_size=128)
    print(len(train_dataset))
    print(len(val_dataset))
    for pic, mask in tqdm(train_loader):
        print(pic.shape, mask.shape)
