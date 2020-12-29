import cv2
import numpy as np
def load_tiff(pic_path, n_channel):
    '''
    load tiff and return numpy
    n_channel:3 for pic and 2 for label pic
    '''
    data = cv2.imread(pic_path, n_channel)
    return data

def crop_data(pic, label, crop_size=256, target_name=None,step_size=10):
    '''
    pic:a numpy with 500*600*3
    label:a numpy with 500*600 with value of 1 and 0
    '''
    h,w,c = pic.shape
    hl, wl = label.shape
    if h < crop_size or w< crop_size:
        print("-------input pic size too small---------")
    elif h!=hl or w!= wl:
        print("-------pic and label size not compact---------")
    else:
        for i in range(0, h-256, step_size):
            for j in range(0, w-256, step_size):
                crop_data = pic[i:i+256, j:j+256, :]
                label_data = label[i:i+256, j:j+256]
                if target_name:
                    dst_name = target_name+"_" + str(i)+"_"+str(j) +".jpg"
                    label_name = target_name+"_"+ str(i)+"_"+str(j) + "_label.jpg"
                else:
                    dst_name =   str(i)+"_"+str(j) +".jpg"
                    label_name = str(i)+"_"+str(j)+ "_label.jpg"
                cv2.imwrite(dst_name, crop_data)
                cv2.imwrite(label_name, label_data)
if __name__ == "__main__":
    import os
    for i in os.listdir():
        if (not i.endswith("ce.tif")) and i.endswith(".tif"):
            data = load_tiff(i, 3)
            label = load_tiff(i.split(".")[0] + "_reference.tif",2)
            crop_data(data,label,target_name="dataSet/crop/" + i.split(".")[0], step_size=30)
