import torch
from models.unet import UNet
import cv2
import matplotlib.pyplot as plt
from utils.transform import Normalize
from utils.transform import ToTensor,Compose
import numpy as np
import torch.nn.functional as f
from skimage import morphology

mean = [74.8559803, 79.1187336, 80.7307415]
std = [19.19655189, 19.56021428, 24.39020428]
model = UNet()
ckpt = torch.load("unet_epoch-400.pth.tar")
model.load_state_dict(ckpt['state_dict'])

data = cv2.imread("oridata/src/Data1.tif", 3).astype(float)

def test_pic(data, model):
    compose = Compose([ToTensor(), Normalize(mean, std)])
    label = np.zeros((500,600))
    data, label = compose(data,label)
    data = f.pad(data,(84,84,6,6), 'constant', 0)
    label = torch.zeros((512,768))
    for i in [0,256]:
        for j in [0,256,512]:
            test_data = data[:,i:i+256,j:j+256]
            test_data
            test_data= test_data.unsqueeze(dim=0)
            label[i:i+256, j:j+256]= model(test_data)[0]
    label = np.round( label.detach().numpy()).astype(np.uint8)
    plt.subplot(141)
    plt.imshow(label[6:506,84:684])
    plt.subplot(142)
    plt.imshow(cv2.imread("oridata/src/Data1.tif", 3))
    plt.subplot(143)
    
    # label_deal = cv2.morphologyEx(label[6:506,84:684], cv2.MORPH_CLOSE, kernel)

    label_deal = morphology.remove_small_objects(label[6:506,84:684], 50)
    label_deal = morphology.remove_small_holes(label_deal, 130)
    kernel = np.ones((5, 5),np.uint8)
    # label_deal = cv2.morphologyEx(label_deal, cv2.MORPH_OPEN, kernel)
    # kernel = np.ones((3, 3),np.uint8)
    label_deal = cv2.morphologyEx(label_deal, cv2.MORPH_CLOSE, kernel)

    
    plt.imshow(label_deal)
    plt.subplot(144)
    data= cv2.imread("oridata/label/Data1_reference.tif", 2).astype(np.uint8)
    plt.imshow(data)
    
test_pic(data, model)
plt.show()
