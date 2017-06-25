import cv2
import numpy as np
import torch.nn as nn
import torch
import argparse
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.transforms import Normalize, Compose, Lambda
from dataset.PlanetDataset import PlanetDataset
import glob
import models.resnet_planet as resnet_planet
from kaggle_util import predict, f2_score, pred_csv
parser = argparse.ArgumentParser(description='Pytorch Planet model test')
parser.add_argument('-c','--checkpoint',default = '/mnt/lvm/zuoxin/kaggle/Planet/0624/resnet18_d_model_best.pth.tar', \
    type = str, help = 'path of the checkpoint to test')
parser.add_argument('-t','--test_dir',default='/mnt/lvmhdd1/dataset/Planet/test-jpg/', \
    type = str, help = 'path containing test images')
parser.add_argument('-tc','--test_csv',default = 'sample_submission_v2.csv',type =str, help = 'test csv file contain test file name')
parser.add_argument('-a','--model_arch',default = 'resnet50_d',type  = str, help = 'model arch to test')
args = parser.parse_args()



class Default(object):
    def __call__(self,img):
        return img
default = Default()
class Rotate90(object):
    def __call__(self,img):
        img = img.rotate(90)
        return img
rotate90 = Rotate90()

class Rotate180(object):
    def __call__(self,img):
        img = img.rotate(180)
        return img
rotate180 = Rotate180()

class Rotate270(object):
    def __call__(self,img):
        img = img.rotate(270)
        return img
rotate270 = Rotate270()
class HorizontalFLip(object):
    def __call__(self, img):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img
horizontalFlip = HorizontalFLip()
class VerticalFLip(object):
    def __call__(self, img):
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        return img
verticalFlip = VerticalFLip()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
threshold = 0.2
# threshold = [0.23166666666666666, 0.19599999999999998, 0.18533333333333335,
#              0.08033333333333334, 0.20199999999999999, 0.16866666666666666,
#              0.20533333333333334, 0.27366666666666667, 0.2193333333333333,
#              0.21299999999999999, 0.15666666666666665, 0.096666666666666679,
#               0.21933333333333335, 0.058666666666666673, 0.19033333333333333,
#               0.25866666666666666, 0.057999999999999996]          # resnet-152

# threshold = [ 0.18533333,  0.18866667, 0.13533333, 0.03633333, 0.221, 0.17666667,
#               0.231, 0.23933333, 0.21966667, 0.169, 0.23333333, 0.21833333,
#               0.24033333, 0.112, 0.40233333, 0.31833333, 0.237]   # densenet-161

# threshold = [ 0.17733333, 0.213, 0.15766667, 0.049, 0.28733333, 0.18066667,
#               0.19666667, 0.212, 0.21566667, 0.17233333, 0.16466667, 0.274,
#               0.27833333, 0.10266667, 0.293, 0.241, 0.08366667]   # densenet-161 + resnet-152

#threshold = [ 0.142, 0.17, 0.122, 0.054, 0.188, 0.156, 0.228, 0.234, 0.142, 0.226,
 #             0.188, 0.192, 0.192, 0.084, 0.242, 0.4, 0.126]      # densenet-161 + densenet-169 + resnet-152

# threshold = [0.136, 0.236, 0.144, 0.044, 0.226, 0.152, 0.214, 0.218, 0.162, 0.204,
#              0.194, 0.19, 0.234, 0.066, 0.236, 0.188, 0.106]        # densenet121+densenet161+densenet169+resnet152

transformations = [default, rotate90, rotate180, rotate270, verticalFlip, horizontalFlip]

models = {# resnet18_planet, resnet34_planet, resnet50_planet, densenet121, densenet161, densenet169
          #  resnet152_planet
            # densenet121,
            'resnet50':'/mnt/lvm/zuoxin/kaggle/Planet/0624/resnet18_d_model_best.pth.tar'
        }


# save probabilities to files for debug
def probs(dataloader):
    """
    returns a numpy array of probabilities (n_transforms, n_models, n_imgs, 17)
    use transforms to find the best threshold
    use models to do ensemble method
    """
    n_transforms = len(transforms)
    n_models = len(models)
    n_imgs = dataloader.dataset.num
    imgs = dataloader.dataset.images.copy()
    probabilities = np.empty((n_transforms, n_models, n_imgs, 17))
    for t_idx, transform in enumerate(transforms):
        t_name = str(transform).split()[1]
        dataloader.dataset.images = transform(imgs)
        for m_idx, model in enumerate(models):
            name = str(model).split()[1]
            net = model().cuda()
            net = nn.DataParallel(net)
            net.load_state_dict(torch.load('models/{}.pth'.format(name)))
            net.eval()
            # predict
            m_predictions = predict(net, dataloader)

            # save
            np.savetxt(X=m_predictions, fname='probs/{}_{}.txt'.format(t_name, name))
            probabilities[t_idx, m_idx] = m_predictions
    return probabilities


def find_best_threshold(labels, probabilities):
    threshold = np.zeros(17)
    acc = 0
    # iterate over transformations
    for t_idx in range(len(transforms)):
        # iterate over class labels
        t = np.ones(17) * 0.15
        selected_preds = probabilities[t_idx]
        selected_preds = np.mean(selected_preds, axis=0)
        best_thresh = 0.0
        best_score = 0.0
        for i in range(17):
            for r in range(500):
                r /= 500
                t[i] = r
                preds = (selected_preds > t).astype(int)
                score = f2_score(labels, preds)
                if score > best_score:
                    best_thresh = r
                    best_score = score
            t[i] = best_thresh
            print('Transform index {}, score {}, threshold {}, label {}'.format(t_idx, best_score, best_thresh, i))
        print('Transform index {}, threshold {}, score {}'.format(t_idx, t, best_score))
        threshold = threshold + t
        acc += best_score
    print('AVG ACC,', acc/len(transforms))
    threshold = threshold / len(transforms)
    return threshold


def get_validation_loader():
    validation = KgForestDataset(
        split='validation-3000',
        transform=Compose(
            [
                Lambda(lambda x: toTensor(x)),
                Normalize(mean=mean, std=std)
            ]
        ),
        height=256,
        width=256
    )
    valid_dataloader = DataLoader(validation, batch_size=256, shuffle=False)
    return valid_dataloader


def get_test_dataset():
    dataset = PlanetDataset(csv_file = args.test_csv, root_dir = args.test_dir,test = True, \
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ]))

    test_loader = DataLoader(dataset, batch_size=256, shuffle=False, pin_memory=True)
    return test_dataloader


def do_thresholding(names, labels):
    preds = np.empty((len(transforms), len(models), 3000, 17))
    print('filenames', names)
    for t_idx in range(len(transforms)):
        for m_idx in range(len(models)):
            preds[t_idx, m_idx] = np.loadtxt(names[t_idx + m_idx])
    t = find_best_threshold(labels=labels, probabilities=preds)
    return t


def get_files(excludes=['resnet18']):
    file_names = glob.glob('probs/*.txt')
    names = []
    for filename in file_names:
        if not any([exclude in filename for exclude in excludes]):
            names.append(filename)
    return names
def load_model(model_name,model_path):
    '''
    model_name: net arc of model
    model_path: checkpoint path
    '''
    checkpoint = torch.load('/mnt/lvm/zuoxin/kaggle/Planet/0624/resnet18_d_model_best.pth.tar')
    model =resnet_planet.PlanetNet(model_name).cuda()
    model =nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model 

def predict_test(t):
    preds = np.zeros((61191, 17))
    # imgs = test_dataloader.dataset.images.copy()
    # iterate over models
    for model, model_path in enumerate(models):
        net = load_model(model_path,model)
        # iterate over transformations
        for transformation in transformations:
            # imgs = transformation(imgs)
            dataset = PlanetDataset(csv_file = args.test_csv, root_dir = args.test_dir,test = True, \
                transform=transforms.Compose([
                transforms.RandomSizedCrop(224),
                transformation,
                transforms.ToTensor(),
                normalize,
                ]))
            test_dataloader = DataLoader(dataset, batch_size=256, shuffle=False, pin_memory=True)
            pred = predict(dataloader=test_dataloader, net=net)
            preds = preds + pred

    preds = preds/(len(models) * len(transforms))
    # preds = preds / len(models)
    pred_csv(predictions=preds, threshold=t, name='resnet50_multi')


if __name__ == '__main__':
    # valid_dataloader = get_validation_loader()
    #test_dataloader = get_test_dataloader()

    # save results to files
    # probabilities = probs(valid_dataloader)

    # get threshold
    # file_names = get_files(['resnet18', 'resnet50', 'resnet34'])
    # t = do_thresholding(file_names, valid_dataloader.dataset.labels)
    # print(t)

    # testing
    predict_test(threshold)
