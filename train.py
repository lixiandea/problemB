# from MyDataset import MyDataset
from models.unet import UNet
import cv2
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
# from logger import Logger
from loss import DiceLoss,CELDice
from torch.utils.data import random_split
import torch
import glob
import os
import sys
import numpy as np 
import matplotlib
import torch.nn as nn
from metric import SegmentationMetric
matplotlib.use('Agg')
from utils import transform
from utils.MyDataset import MyDataset
import matplotlib.pyplot as plt
from models.multimodel import R2U_Net, AttU_Net,R2AttU_Net,RAUNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


mean = [74.8559803, 79.1187336, 80.7307415]
# mean = [0.485, 0.456, 0.406]
# mean = [item * value_scale for item in mean]
# std = [0.229, 0.224, 0.225]
std = [19.19655189, 19.56021428, 24.39020428]
# std = [item * value_scale for item in std]
loss_train = []
loss_valid = []

#def test(model, pic_path, label_path):
#    model.eval()
#    data = (cv2.imread(pic_path, 3) - mean )/ std
#    label = cv2.imread(label_path, 2)
#    acc = 0
#    mIoU = 0
#    FWIoU = 0
#    for i in [0, 240]:
#        for j in [0, 256]:
#            metric = SegmentationMetric(2)
#            test_data = data[i:i+256,j:j+256, :].transpose(2,0,1)
#            test_data = test_data[np.newaxis, :, :, :].astype(np.float32)
#            test_label = label[i:i+256,j:j+256]
#            test_data = torch.from_numpy(test_data).cuda()
#            res = model(test_data)
#            res = np.around(res.detach().cpu().numpy()[0,0,:,:]).astype(np.uint8)
#            # print(res)
#            #plt.figure()
#            #plt.imshow(res)
#            #plt.figure()
#            #plt.imshow(test_label)
#            metric.addBatch(res, test_label)
#            acc += metric.pixelAccuracy()
#            mIoU += metric.meanIntersectionOverUnion()
#            FWIoU += metric.Frequency_Weighted_Intersection_over_Union()
#            # print(acc, mIoU, FWIoU)
#    return acc, mIoU, FWIoU

    


# def train(model, optimizer, dataset, save_epoch=100, test_pic="oridata/src/Data8.tif", test_label = "oridata/label/Data8_reference.tif", epoch_num=200):
def train(model, optimizer, dataset, save_epoch=100,  epoch_num=200):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # best_validation_dsc = 0.0
    dsc_loss = DiceLoss()
    # dsc_loss = nn.BCELoss()
    # dsc_loss = CELDice(dice_weight=0.2)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=1, batch_size=2)
    val_loader = DataLoader(val_dataset, shuffle=False, num_workers=1, batch_size=2)
    print(len(train_dataset))
    print(len(val_dataset))
    loaders = {"train": train_loader, "valid": val_loader}
    

    step = 0
    # accs = []
    # mIoUs = []
    # FWIoU = []
    for epoch in tqdm(range(epoch_num)):

        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            validation_pred = []
            validation_true = []

            for i, data in enumerate(loaders[phase]):
                if phase == "train":
                    step += 1

                x, y_true = data
                x, y_true = x.to(device), y_true.long().to(device)
                # y_true = y_true.squeeze(1)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_pred = torch.sigmoid(model(x))

                    loss = dsc_loss(y_pred, y_true)

                    if phase == "valid":
                        epoch_val_loss += loss.item()
                        # loss_valid.append(loss.item())
                        y_pred_np = y_pred.detach().cpu().numpy()
                        validation_pred.extend(
                            [y_pred_np[s] for s in range(y_pred_np.shape[0])]
                        )
                        y_true_np = y_true.detach().cpu().numpy()
                        validation_true.extend(
                            [y_true_np[s] for s in range(y_true_np.shape[0])]
                        )
                        
                    if phase == "train":
                        epoch_train_loss += loss.item()
                        # loss_train.append(epoch_train_loss)
                        loss.backward()
                        optimizer.step()
            if phase == "train":
                loss_train.append(epoch_train_loss)
            else:
                loss_valid.append(epoch_val_loss)
        plt.cla()
        plt.plot(loss_train, color='r', label='train_loss')
        plt.plot(loss_valid, color='b', label='val_loss')
        plt.legend()
        plt.savefig(save_dir +'/loss.png' ,format='png')
        # print("train_loss: ", epoch_train_loss, "val_loss: ", epoch_train_loss)
                    # logger.scalar_summary("val_dsc", mean_dsc, step)
        
        # epoch % save_epoch == (save_epoch - 1):
        # torch.save({
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'opt_dict': optimizer.state_dict(),
        # }, os.path.join(save_dir, 'models', "unet" + '_epoch-' + str(epoch +1) + '.# pth.tar'))
        # print("Save model at {}\n".format(os.path.join(save_dir, 'models', "unet" + # '_epoch-' + str(epoch + 1) + '.pth.tar')))

        #acc, mIoU, fIou = test(model, test_pic, test_label)    
        #accs.append(acc)
        #mIoUs.append(mIoU)
        #FWIoU.append(fIou)
        #plt.cla()
        #plt.plot(accs, color='r', label='acc')
        #plt.plot(mIoUs, color='g', label='IOU')
        #plt.plot(FWIoU, color='b', label='FIOU')
        #plt.legend()
        #plt.savefig(save_dir +'/metrics.png' ,format='png')
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'opt_dict': optimizer.state_dict(),
        'train_loss': loss_train,
        'val_loss': loss_valid,
        }, os.path.join(save_dir, 'models', sys.argv[1] + '-final'+ '.pth.tar'))
    print("Save final model at {} ".format(os.path.join(save_dir, 'models', "unet" + '-final'+ '.pth.tar')))
        # print("Best validation mean DSC: {:4f}".format(best_validation_dsc))
if __name__=="__main__":
    if not os.path.exists("run"):
        os.mkdir("run")
    runs = sorted(glob.glob(os.path.join( 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
    save_dir = os.path.join('run', 'run_' + '0' * (2 - len(str(run_id)))+ str(run_id))
    os.mkdir(save_dir)
    os.mkdir(save_dir+"/models")
    print("save_dir: ",save_dir )

 # R2U_Net, AttU_Net,R2AttU_Net,RAUNet
    if sys.argv[1] == 'unet':
        model = UNet().to(device)
    if sys.argv[1] == 'R2U_Net':
        model = R2U_Net().to(device)
    if sys.argv[1] == 'AttU_Net':
        model = AttU_Net().to(device)
    if sys.argv[1] == 'R2AttU_Net':
        model = R2AttU_Net().to(device)
    if sys.argv[1] == 'RAUNet':
         model = RAUNet().to(device)
    
    # model = R2U_Net().to(device)




    train_transform = transform.Compose([
    # transform.RandScale([args.scale_min, args.scale_max]),
    transform.RandRotate([0, 45], padding=mean, ignore_label=0),
    transform.RandomGaussianBlur(),
    transform.RandomHorizontalFlip(),
    transform.Crop([256, 256], crop_type='rand', padding=mean, ignore_label=0),
    transform.ToTensor(),
    transform.Normalize(mean=mean, std=std)
    ])
    dataset = MyDataset("dataset/train", transform=train_transform)
    # train(unet, optimizer, dataset, epoch_num=int(sys.argv[1]))
    for i in range(10):
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        train(model, optimizer, dataset, epoch_num=int(sys.argv[2]))
        # train(model, optimizer, dataset, epoch_num=10)



