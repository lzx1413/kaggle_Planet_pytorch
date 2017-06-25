'''
Training script for ImageNet
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import models.imagenet as customized_models
import models.resnet_planet as resnet_planet
from dataset.PlanetDataset import PlanetDataset
from PIL import Image

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from kaggle_util import *
import numpy as np

# Models
default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch Planet multilabe classification Training')

# Datasets
parser.add_argument('-d', '--data', default='/mnt/lvmhdd1/dataset/Planet/train-jpg/', type=str)
parser.add_argument('-t', '--train_csv', default='dataset/planet_train.txt', type=str)
parser.add_argument('-v', '--val_csv', default='dataset/planet_val.txt', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=64, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=64, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[10,20],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='/mnt/lvm/zuoxin/kaggle/Planet/0624', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
#parser.add_argument('--resume', default='/mnt/lvm/zuoxin/kaggle/Planet/0624/model_best.pth.tar', type=str, metavar='path',
 #help='path to latest checkpoint (default: none)') # Architecture
parser.add_argument('--resume', default='', type=str, metavar='path',
 help='path to latest checkpoint (default: none)') # Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

#Device options
parser.add_argument('--ngpu', default=2, type=int,
                    help='number of GPUs to use for training')
parser.add_argument('--gpu_id', default='1', type=str,
help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

min_loss = np.inf  # least test loss

class RandomVerticalFLip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        return img

def lr_scheduler(epoch, optimizer):
    if epoch <= 10:
        lr = 1e-1
    elif 10 < epoch <= 30:
        lr = 1e-2
    elif 30 < epoch <= 45:
        lr = 5e-3
    else:
        lr = 1e-3

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_optimizer(model, pretrained=True, lr=5e-5, weight_decay=5e-5):
    if pretrained:
        params = [
            {'params': model.features.parameters(), 'lr': lr},
            {'params': model.classifier.parameters(), 'lr': lr * 10}
        ]
    else:
        params = [
            {'params': model.features.parameters(), 'lr': lr},
            {'params': model.classifier.parameters(), 'lr': lr}
        ]
    return optim.Adam(params=params, weight_decay=weight_decay)

def main():
    global  min_loss
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
         PlanetDataset(csv_file = args.train_csv, root_dir = args.data,transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            RandomVerticalFLip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
         PlanetDataset(csv_file = args.val_csv, root_dir = args.data,transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)
        
    # create model
    '''
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    elif args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    baseWidth=args.base_width,
                    cardinality=args.cardinality,
                )
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()
    '''
    model = resnet_planet.PlanetNet()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # define loss function (criterion) and optimizer
    criterion = nn.MultiLabelSoftMarginLoss().cuda()
    ignored_params = list(map(id, model.fc.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params,
                     model.parameters())
    params = [
            {'params': base_params, 'lr': args.lr},
            {'params': model.fc.parameters(), 'lr': args.lr * 10}
    ]
    optimizer = optim.SGD(params = params, momentum=args.momentum, weight_decay=args.weight_decay,nesterov = True)
    model = torch.nn.DataParallel(model).cuda()

    # Resume
    title = 'Planet-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        min_loss = checkpoint['min_loss']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=False)
    else:
        pass
        logger = Logger(fpath = os.path.join(args.checkpoint, 'log.txt'),title = title,resume = False)
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss','Train F2','Valid F2'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss = test(val_loader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss,train_f2 = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        test_loss,test_f2 = test(val_loader, model, criterion, epoch, use_cuda)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss,train_f2,test_f2])

        # save model
        is_best = test_loss < min_loss
        min_loss = min(test_loss, min_loss)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'min_loss': min_loss,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint,filename = title+'_'+str(epoch))

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('min loss:')
    print(min_loss)

def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_losses = AverageMeter()
    end = time.time()
    train_f2 = AverageMeter()

    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        binary_out = F.sigmoid(outputs)
        binary_out[binary_out>=0.2] = 1
        binary_out[binary_out<0.2] = 0
        train_f2.update(f2_score(binary_out.data.cpu().numpy(),targets.data.cpu().numpy()))
        train_losses.update(loss.data[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | F2: {f2_score}'.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=train_losses.avg,
                    f2_score = train_f2.avg
                    )
        bar.next()
    bar.finish()
    return (train_losses.avg,train_f2.avg)

def test(val_loader, model, criterion, epoch, use_cuda):
    global min_loss

    batch_time = AverageMeter()
    data_time = AverageMeter()
    test_losses = AverageMeter()
    test_f2 = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        test_losses.update(loss.data[0], inputs.size(0))
        binary_out = F.sigmoid(outputs)
        binary_out[binary_out>=0.2] = 1
        binary_out[binary_out<0.2] = 0
        test_f2.update(f2_score(binary_out.data.cpu().numpy(),targets.data.cpu().numpy()))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | F2: {f2_score}'.format(
                    batch=batch_idx + 1,
                    size=len(val_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=test_losses.avg,
                    f2_score = test_f2.avg
                    )
        bar.next()
    bar.finish()
    return (test_losses.avg,test_f2.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint'):
    filepath = os.path.join(checkpoint, filename+'.pth.tar')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'resnet18_d_model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()