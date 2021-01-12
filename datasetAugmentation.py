#coding=utf-8
import cv2
import random
import os
import numpy as np
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt

img_w = 256  
img_h = 256  
# 1-7 for train and 8 for test
image_sets = ["Data" + str(i) +".tif" for i in range(1,9)]

def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)
    

def rotate(xb,yb,angle):
    M_rotate = cv2.getRotationMatrix2D((img_w/2, img_h/2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb,yb
    
def blur(img):
    img = cv2.blur(img, (3, 3));
    return img

def add_noise(img):
    for i in range(200): #添加点噪声
        temp_x = np.random.randint(0,img.shape[0])
        temp_y = np.random.randint(0,img.shape[1])
        img[temp_x][temp_y] = 255
    return img
    
    
def data_augment(xb,yb):
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,90)
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,180)
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,270)
    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转
        yb = cv2.flip(yb, 1)
        
    # if np.random.random() < 0.25:
    #     xb = random_gamma_transform(xb,1.0)
    #     
    # if np.random.random() < 0.25:
    #     xb = blur(xb)
    # 
    # if np.random.random() < 0.2:
    #     xb = add_noise(xb)
        
    return xb,yb

def creat_dataset(image_num = 100000, mode = 'original'):
    print('creating dataset...')
    image_each = image_num / len(image_sets)
    g_count = 0
    for i in tqdm(range(len(image_sets))):
        count = 0
        src_img = image = cv2.normalize(cv2.imread(os.path.join('oridata', 'src',image_sets[i]) , 3), None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3).astype(np.uint8)  # 3 channels
        label_img = cv2.imread(os.path.join('oridata', 'label', image_sets[i].split('.')[0] + "_reference.tif"), 2)  # single channel
        X_height,X_width,_ = src_img.shape
        while count < image_each:
            random_width = random.randint(0, X_width - img_w - 1)
            random_height = random.randint(0, X_height - img_h - 1)
            src_roi = src_img[random_height: random_height + img_h, random_width: random_width + img_w,:]
            label_roi = label_img[random_height: random_height + img_h, random_width: random_width + img_w]
            if mode == 'augment':
                src_roi,label_roi = data_augment(src_roi,label_roi)
            
            
            # visualize = label_roi.copy() *255
            plt.cla()
            plt.subplot(121)
            plt.imshow(src_roi)
            plt.subplot(122)
            plt.imshow(label_roi)
            #plt.subplot(133)
            #plt.imshow(visualize)
            plt.savefig(('dataset/visualize/%d.png' % g_count))
            # cv2.imwrite(('dataset/visualize/%d.png' % g_count),visualize)
            cv2.imwrite(('dataset/train/src/%d.png' % g_count),src_roi)
            cv2.imwrite(('dataset/train/label/%d.png' % g_count),label_roi)
            count += 1 
            g_count += 1


            
    

if __name__=='__main__':  
    if not os.path.exists("dataset"):
        os.makedirs('dataset/visualize')
        os.makedirs('dataset/train/src')
        os.makedirs('dataset/train/label')
    # print(sys.argv)
    creat_dataset(mode='augment', image_num=int(sys.argv[1]))
    # creat_dataset(mode='augment', image_num=80)
