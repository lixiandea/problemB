from MyDataset import MyDataset
from models import UNet
import cv2
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
# from logger import Logger
from loss import DiceLoss
from torch.utils.data import random_split
import torch
import glob
import os
import numpy as np 
import matplotlib
from metric import SegmentationMetric
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if not os.path.exists("run"):
    os.mkdir("run")
runs = sorted(glob.glob(os.path.join( 'run', 'run_*')))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
save_dir = os.path.join('run', 'run_' + '0' * (2 - len(str(run_id)))+ str(run_id))
os.mkdir(save_dir)
os.mkdir(save_dir+"/models")
print("save_dir: ",save_dir )

def test(model, pic_path, label_path):
    model.eval()
    data = cv2.imread(pic_path, 3)
    label = cv2.imread(label_path, 2)
    for i in [0, 240]:
        for j in [0, 256]:
            metric = SegmentationMetric(2)
            test_data = data[i:i+256,j:j+256, :].transpose(2,0,1)
            test_data = test_data[np.newaxis, :, :, :].astype(np.float32)
            test_label = label[i:i+256,j:j+256]
            test_data = torch.from_numpy(test_data).cuda()
            res = model(test_data)
            res = np.around(res.detach().cpu().numpy()[0,0,:,:]).astype(np.uint8)
            # print(res)
            #plt.figure()
            #plt.imshow(res)
            #plt.figure()
            #plt.imshow(test_label)
            metric.addBatch(res, test_label)
            acc = metric.pixelAccuracy()
            mIoU = metric.meanIntersectionOverUnion()
            FWIoU = metric.Frequency_Weighted_Intersection_over_Union()
            # print(acc, mIoU, FWIoU)
            return acc, mIoU, FWIoU

    plt.show()


def train(model, optimizer, dataset, save_epoch=100, test_pic="Data8.tif", test_label = "Data8_reference.tif"):
    best_validation_dsc = 0.0
    dsc_loss = DiceLoss()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=4, batch_size=64)
    val_loader = DataLoader(val_dataset, shuffle=False, num_workers=1, batch_size=64)
    print(len(train_dataset))
    print(len(val_dataset))
    loaders = {"train": train_loader, "valid": val_loader}
    
    loss_train = []
    loss_valid = []
    step = 0
    accs = []
    mIoUs = []
    FWIoU = []
    for epoch in tqdm(range(500)):

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
                x, y_true = x.to(device), y_true.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_pred = model(x)

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
        
        if epoch % save_epoch == (save_epoch - 1):
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, 'models', "unet" + '_epoch-' + str(epoch +1) + '.pth.tar'))
            print("Save model at {}\n".format(os.path.join(save_dir, 'models', "unet" + '_epoch-' + str(epoch + 1) + '.pth.tar')))

        acc, mIoU, fIou = test(model, test_pic, test_label)    
        accs.append(acc)
        mIoUs.append(mIoU)
        FWIoU.append(fIou)
        plt.cla()
        plt.plot(accs, color='r', label='acc')
        plt.plot(mIoUs, color='g', label='IOU')
        plt.plot(FWIoU, color='b', label='FIOU')
        plt.legend()
        plt.savefig(save_dir +'/metrics.png' ,format='png')
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'opt_dict': optimizer.state_dict(),
        'acc': accs,
        'mIoUs': mIoUs,
        'FWIoU': FWIoU
        }, os.path.join(save_dir, 'models', "unet" + '-final'+ '.pth.tar'))
    print("Save final model at {} ".format(os.path.join(save_dir, 'models', "unet" + '-final'+ '.pth.tar')))
        # print("Best validation mean DSC: {:4f}".format(best_validation_dsc))
if __name__=="__main__":
    
    unet = UNet().to(device)
    dataset = MyDataset("dataSet/train")
    optimizer = optim.Adam(unet.parameters(), lr=1e-3)
    train(unet, optimizer, dataset)