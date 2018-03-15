import os
from os import listdir
from os.path import join
import random
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Scale
import numpy as np
import skimage.io
import skimage.transform


def generate_LRImage(HR_Dir='HRDataSet',LR_Dir='LRDataSet',scale=3):

    if os.path.exists(LR_Dir):
        print('LRDataSet exists')
        return

    os.mkdir(LR_Dir)

    imgs = listdir(HR_Dir)
    for file in imgs:
        if is_image_file(file):
            fu_path = join(HR_Dir,file)
            img = skimage.io.imread(fu_path)
            img = modcrop(img,scale)
            w,h,c = img.shape
            img = skimage.transform.resize(img,(w/scale,h/scale))
            skimage.io.imsave(join(LR_Dir,'LR'+file),img)

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



def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.bmp'])

class TrainDatasetFromFolder(Dataset):
    def __init__(self, HR_dir, LR_dir, lr_size=33,hr_size=21, upscale_factor=3):
        super(TrainDatasetFromFolder, self).__init__()
        self.lr_size = lr_size
        self.hr_size = hr_size
        self.s = upscale_factor
        self.HRimage_filenames = [join(HR_dir, x) for x in listdir(HR_dir) if is_image_file(x)]
        self.LRimage_filenames = [join(LR_dir, x) for x in listdir(LR_dir) if is_image_file(x)]


    def __getitem__(self, index):
        hr_image = Image.open(self.HRimage_filenames[index])
        lr_image = Image.open(self.LRimage_filenames[index])
        w,h = lr_image.size
        w = self.s *w
        h = self.s * h
        lr_image.resize((w,h),Image.CUBIC)
        p = (self.lr_size)/2
        x1 = random.randint(p, w - p)
        y1 = random.randint(p, h - p)
        LRsub_pix = lr_image.crop((x1-p,y1-p,x1+p+1,y1+p+1))

        p2 = (self.hr_size)/2
        HRsub_pix = hr_image.crop((x1-p2,y1-p2,x1+p2+1,y1+p2+1))

        return ToTensor()(LRsub_pix), ToTensor()(HRsub_pix)

    def __len__(self):
        return len(self.HRimage_filenames)
