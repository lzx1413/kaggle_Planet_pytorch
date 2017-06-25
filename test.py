#coding:utf-8
import os
import argparse
import torch.nn as nn
from torch.nn import functional as F
import models.resnet_planet as resnet_planet
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
import torch
from dataset.PlanetDataset import PlanetDataset
from torch.utils.data.dataloader import DataLoader
from kaggle_util import pred_csv, predict
import models.resnet_planet as resnet_planet

parser = argparse.ArgumentParser(description='Pytorch Planet model test')
parser.add_argument('-c','--checkpoint',default = '/mnt/lvm/zuoxin/kaggle/Planet/0624/resnet18_d_model_best.pth.tar', \
    type = str, help = 'path of the checkpoint to test')
parser.add_argument('-t','--test_dir',default='/mnt/lvmhdd1/dataset/Planet/test-jpg/', \
    type = str, help = 'path containing test images')
parser.add_argument('-tc','--test_csv',default = 'sample_submission_v2.csv',type =str, help = 'test csv file contain test file name')
parser.add_argument('-a','--model_arch',default = 'resnet50_d',type  = str, help = 'model arch to test')
args = parser.parse_args()

#BEST_THRESHOLD = np.array([0.2071, 0.1986, 0.1296, 0.0363, 0.2355 , 0.1766, 0.2255, 0.257, 0.1922,
#                            0.1462, 0.2676, 0.0931, 0.213, 0.1041, 0.2606, 0.2872, 0.151])
BEST_THRESHOLD = 0.2
class Rotate90(object):
    def __call__(self,img):
        img = img.rotate(90)
        return img
rotate90 = Rotate90()


def load_model(model_name,model_path):
    '''
    model_name: net arc of model
    model_path: checkpoint path
    '''
    checkpoint = torch.load(model_path)
    model =resnet_planet.PlanetNet().cuda()
    model =nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model 


def test():
    net = load_model(args.model_arch,args.checkpoint)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = PlanetDataset(csv_file = args.test_csv, root_dir = args.test_dir,test = True, \
        transform=transforms.Compose([
            rotate90,
            transforms.ToTensor(),
            normalize,
            ]))

    test_loader = DataLoader(dataset, batch_size=256, shuffle=False, pin_memory=True)
    probs = predict(net, test_loader)

    # probs = np.empty((61191, 17))
    # current = 0
    # for batch_idx, (images, im_ids) in enumerate(test_loader):
    #     num = images.size(0)
    #     previous = current
    #     current = previous + num
    #     logits = net(Variable(images.cuda(), volatile=True))
    #     prob = F.sigmoid(logits)
    #     probs[previous:current, :] = prob.data.cpu().numpy()
    #     print('Batch Index ', batch_idx)

    pred_csv(probs, name=args.model_arch, threshold=BEST_THRESHOLD,csv_name = args.test_csv)


if __name__ == '__main__':
    test()