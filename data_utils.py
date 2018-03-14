import os
from os import listdir
from os.path import join
import random
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Scale
import numpy as np
import skimage.io

def generate_LRImage(h_Dir):
    imgs = listdir(h_Dir)

    os.mkdir('LRDataSet')
    for file in imgs:
        if is_image_file():
            fu_path = join(h_Dir,file)
            img = skimage.io.imread(fu_path)

def modcrop(image, scale=3):
    """
    To scale down and up the original image, first thing to do is to have no remainder while scaling operation.

    We need to find modulo of height (and width) and scale factor.
    Then, subtract the modulo from height (and width) of original image size.
    There would be no remainder even after scaling operation.
    """
    if len(image.shape) == 3:
        h, w, _ = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w]
    return image

def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Scale(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])



def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.bmp'])

class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, lr_size=33,hr_size=21, upscale_factor=3):
        super(TrainDatasetFromFolder, self).__init__()
        self.lr_size = lr_size
        self.hr_size = hr_size
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(lr_size, upscale_factor)
        self.hr_transform = train_hr_transform(hr_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        w,h = hr_image.size
        x1 = random.randint(0, w - self)
        y1 = random.randint(0, h - th)
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)
