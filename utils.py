import numpy as np
import math
import torchvision
from scipy.misc import imresize, imsave
import os
import config
def set_requires_grad(module , b ):
    for parm in module.parameters():
        parm.requires_grad = b


def tensor2im(img, imtype=np.uint8, unnormalize=False, idx=0, nrows=None):
    # select a sample or create grid if img is a batch
    if len(img.shape) == 4:
        nrows = nrows if nrows is not None else int(math.sqrt(img.size(0)))
        img = img[idx] if idx >= 0 else torchvision.utils.make_grid(img, nrows)

    img = img.cpu().float()
    if unnormalize:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        for i, m, s in zip(img, mean, std):
            i.mul_(s).add_(m)

    image_numpy = img.detach().numpy()
    image_numpy_t = np.transpose(image_numpy, (1, 2, 0))
    image_numpy_t = image_numpy_t*255.0

    return image_numpy_t.astype(imtype)


def tensor2maskim(mask, imtype=np.uint8, idx=0, nrows=1):
    im = tensor2im(mask, imtype=imtype, idx=idx, unnormalize=False, nrows=nrows)
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=-1)
    return im

def save_img(img, file_name, imtype=np.uint8, unnormalize=False, idx=0, nrows=None):
    # select a sample or create grid if img is a batch
    if len(img.shape) == 4:
        nrows = nrows if nrows is not None else int(math.sqrt(img.size(0)))
        img = img[idx] if idx >= 0 else torchvision.utils.make_grid(img, nrows)

    img = img.cpu().float()
    if unnormalize:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        for i, m, s in zip(img, mean, std):
            i.mul_(s).add_(m)

    image_numpy = img.detach().numpy()
    image_numpy_t = np.transpose(image_numpy, (1, 2, 0))
    image_numpy_t = image_numpy_t*255.0
    image_numpy_t = imresize(image_numpy_t, (250, 200, 3))
    if not os.path.exists(config.test['result_path']):
        os.mkdir(config.test['result_path'])
    imsave(os.path.join(config.test['result_path'], file_name), image_numpy_t)
    return image_numpy_t.astype(imtype)