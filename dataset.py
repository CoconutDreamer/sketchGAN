import torch
import scipy.io as sio
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from os import listdir
from os.path import join
from scipy.misc import imread, imresize, imsave
from scipy.io import loadmat
from PIL import Image
import random
import config
from skimage import color
import matplotlib.pyplot as plt
### image w x h
image_sz = 256

### dict w x h
dict_sz = 286


### Load data set
# read data from data set files
class dataFromFolder(data.Dataset):
    def __init__(self, data_dir, opt):
        super(dataFromFolder, self).__init__()
        self.photo_path = join(data_dir, "Photos")
        self.sketches_path = join(data_dir, "Sketches")
        self.face_path = join(data_dir, "Faces")

        self.image_file_names = [x for x in listdir(self.photo_path) if isImageFile(x)]
        self.opt = opt

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        # a batch include img, mask, and face GT without background.
        batch = {}
        w = random.randint(0, max(0, dict_sz - image_sz - 1))
        h = random.randint(0, max(0, dict_sz - image_sz - 1))
        t = random.random()
        img_name = self.image_file_names[idx]
        photo = loadImage(join(self.photo_path, img_name), w, h, t, self.opt)
        sketch = loadSketch(join(self.sketches_path, img_name), w, h, t, self.opt)
        face = loadSketch(join(self.face_path, img_name[:-3] + 'png'), w, h, t, self.opt)

        batch['photo'] = photo
        batch['sketch'] = sketch
        batch['face'] = face

        return batch

### get train data
def getTrainData(root_dir, opt='Training'):
    train_dir = join(root_dir, "Training")
    return dataFromFolder(train_dir, opt)

### get test data
def getTestData(root_dir, opt='Testing'):
    test_dir = join(root_dir, "Testing")
    return dataFromFolder(test_dir, opt)
### Image operatoration
### photo
# load image from data set
def loadImage(file_path, w, h, t, opt):
    img = Image.open(file_path).convert('L')
    totensor = transforms.ToTensor()
    if opt == 'Training':
        img = img.resize((dict_sz, dict_sz), Image.LANCZOS)

        if t < 0.5:  # flip
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # gray scale
        '''img = np.array(img)
        img_lab = color.rgb2lab(img / 255)
        img = img_lab[:, :, 0:1]'''
        img = totensor(img)
        img = img[:, h: h + image_sz,
              w: w + image_sz]

    else:  # 'Testing'
        img = img.resize((image_sz, image_sz), Image.LANCZOS)
        img = totensor(img)

    return img


def loadSketch(file_path, w, h, t, opt):
    img = Image.open(file_path).convert('RGB')
    totensor = transforms.ToTensor()
    if opt == 'Training':
        img = img.resize((dict_sz, dict_sz), Image.LANCZOS)

        if t < 0.5:  # flip
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        img = totensor(img)
        img = img[:, h: h + image_sz,
              w: w + image_sz]

    else:  # 'Testing'
        img = img.resize((image_sz, image_sz), Image.LANCZOS)
        img = totensor(img)

    return img

### is the image loaded right
def isImageFile(file_name):
    return any(file_name.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".tif"])


def weight_mask_generate():
    EYE_W, EYE_H = 60, 60
    NOSE_W, NOSE_H = 60, 44
    MOUTH_W, MOUTH_H = 76, 44
    IMG_SIZE = 256

    left_eye = config.loss['weight_pixel_local'] * torch.ones( 1, EYE_H, EYE_W)
    right_eye = config.loss['weight_pixel_local'] * torch.ones( 1, EYE_H, EYE_W)
    nose = config.loss['weight_pixel_local'] * torch.ones( 1, NOSE_H, NOSE_W)
    mouth = config.loss['weight_pixel_local'] * torch.ones( 1, MOUTH_H, MOUTH_W)

    f_left_eye = torch.nn.functional.pad(left_eye, (
    68 - EYE_W // 2 - 1, IMG_SIZE - (68 + EYE_W // 2 - 1), 110 - EYE_H // 2 - 1, IMG_SIZE - (110 + EYE_H // 2 - 1)), value=0)
    f_right_eye = torch.nn.functional.pad(right_eye, (
    172 - EYE_W // 2 - 1, IMG_SIZE - (172 + EYE_W // 2 - 1), 108 - EYE_H // 2 - 1, IMG_SIZE - (108 + EYE_H // 2 - 1)), value=0)
    f_nose = torch.nn.functional.pad(nose, (
    128 - NOSE_W // 2 - 1, IMG_SIZE - (128 + NOSE_W // 2 - 1), 158 - NOSE_H // 2 - 1, IMG_SIZE - (158 + NOSE_H // 2 - 1)), value=0)
    f_mouth = torch.nn.functional.pad(mouth, (
    130 - MOUTH_W // 2 - 1, IMG_SIZE - (130 + MOUTH_W // 2 - 1), 208 - MOUTH_H // 2 - 1,
    IMG_SIZE - (208 + MOUTH_H // 2 - 1)), value=0)
    mask = torch.max(torch.stack([f_left_eye, f_right_eye, f_nose, f_mouth], dim=0), dim=0)[0]

    return mask


class TestDataset(data.Dataset):
    def __init__(self, data_dir):
        super(TestDataset, self).__init__()
        self.photo_path = data_dir
        self.image_file_names = [x for x in listdir(self.photo_path) if isImageFile(x)]
        self.opt='Testing'

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):

        batch = {}
        w = random.randint(0, max(0, dict_sz - image_sz - 1))
        h = random.randint(0, max(0, dict_sz - image_sz - 1))
        t = random.random()
        img_name = self.image_file_names[idx]
        photo = loadImage(join(self.photo_path, img_name), w, h, t, self.opt)
        batch['photo'] = photo

        return batch


