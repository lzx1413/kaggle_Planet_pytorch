from __future__ import print_function, division
import os
import torch
import pandas as pd
import torch.utils.data as data
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    return pil_loader(path)

class PlanetDataset(data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir,image_type='jpg',transform=None,test = False,loader = default_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            image_type (string): image type in the dataset 
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.label_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.image_type = image_type
        self.transform = transform
        self.loader = loader
        self.test = test
        self.num = len(self.label_file)
    def __len__(self):
        return len(self.label_file)

    def __getitem__(self,idx):
        img_name = os.path.join(self.root_dir, self.label_file.ix[idx, 0]+'.'+self.image_type)
        image = self.loader(img_name)
        #landmarks = landmarks.reshape(-1, 2)
        #sample = {'image': image, 'lables': lables}

        if self.transform:
            image = self.transform(image)
        if self.test:
            return image
        labels = self.label_file.ix[idx, 1:].as_matrix().astype('float32')
        return image,labels

if __name__ == '__main__':
    planet_dataset = PlanetDataset(csv_file='planet_train_val.txt',
                                    root_dir='/mnt/lvmhdd1/dataset/Planet/train-jpg/',
                                    transform = transforms.Compose([
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()]))
    dataloader = data.DataLoader(planet_dataset, batch_size=4,
                        shuffle=True, num_workers=4)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched[0].size(),
          sample_batched[1].size())