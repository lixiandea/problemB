
import os
import random
import utils.transform as transform
import numpy as np
import torch
from skimage.io import imread
from torch.utils.data import Dataset
from torch.utils.data import random_split
import cv2  
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
def plot(img, label):
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(label)
    plt.show()

class MyDataset(Dataset):
    """Brain MRI dataset for FLAIR abnormality segmentation"""
    
    in_channels = 3
    out_channels = 1

    def __init__(
        self,
        images_dir,
        image_size=512,
        transform=None
    ):
        # read images
        self.pics = []
        self.masks = []
        self.transform = transform
        pic_dir = os.path.join(images_dir, "src")
        label_dir = os.path.join(images_dir, "label")
        print("------{} pics------".format(len(os.listdir(pic_dir))))
        for i in tqdm(os.listdir(pic_dir)):
        # for i in tqdm(range(10)):
            self.pics.append( np.float32(cv2.imread( os.path.join(pic_dir, i), 3 )))
            self.masks.append(cv2.imread( os.path.join(label_dir, i), 2).astype(float))
        

    def __len__(self):
        return len(self.pics )

    def __getitem__(self, idx):
        pic =  self.pics[idx]
        mask = self.masks[idx]
        if self.transform is not None:
            pic, mask = self.transform(pic, mask)
        return pic, mask

    # def normalization(self, img):
    #     for i in range(3):
    #         img[:,:,i] = (img[:,:,i] - self.mu[i])/self.sigma[i]
    #     return img
def channel_hist(image):
	'''
	画三通道图像的直方图
	'''
	color = ('b', 'g', 'r')   #这里画笔颜色的值可以为大写或小写或只写首字母或大小写混合
	for i , color in enumerate(color):
		hist = cv2.calcHist([image], [i], None, [100], [int(np.min(image)) - 1 , int(np.max(image)) + 1])  #计算直方图
		plt.plot(hist, color)
		plt.xlim([int(np.min(image)) - 1 , int(np.max(image)) + 1])
	plt.show()


if __name__ == "__main__":
    value_scale = 255
    mean = [74.8559803, 79.1187336, 80.7307415]
    # mean = [0.485, 0.456, 0.406]
    # mean = [item * value_scale for item in mean]
    # std = [0.229, 0.224, 0.225]
    std = [19.19655189, 19.56021428, 24.39020428]
    # std = [item * value_scale for item in std]

    train_transform = transform.Compose([
        transform.Resize((512,512)),
        # transform.RandScale([0.5, 2]),
        transform.RandRotate([0, 45], padding=mean, ignore_label=0),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([256, 256], crop_type='rand', padding=mean, ignore_label=0),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)
        ])
    dataset = MyDataset("dataset/train", transform=train_transform)
    batchSize = 16
    validation_split = 2
    shuffle_dataset = True
    random_seed = 42
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=1, batch_size=1)
    val_loader = DataLoader(val_dataset, shuffle=False, num_workers=1, batch_size=1)
    print(len(train_dataset))
    print(len(val_dataset))
    for pic, mask in tqdm(train_loader):
        print(pic.shape, mask.shape)
        pic = pic[0].numpy().transpose((1,2,0))
        # channel_hist(pic)
        plot(pic, mask[0,0].numpy())
        print(np.max(pic),np.mean(pic), np.min(pic))

