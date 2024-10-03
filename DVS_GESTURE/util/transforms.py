import random
import torch
import numpy as np
import torchvision.transforms.functional as F
from typing import Dict
from torchvision import transforms, datasets


def hflip(image):
    flipped_image = F.hflip(image)
    return flipped_image


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return hflip(img)
        return img


def resize(image, size):
    def get_size_with_aspect_ratio(image_size, size):
        h, w = image_size
        if h == size:
            return (h, w)
        oh = size
        ow = int(size * w / h)
        return [oh, ow]
    def get_size(image_size, size):
        return get_size_with_aspect_ratio(image_size, size)
    size = get_size((image.shape[1], image.shape[2]), size)
    rescaled_image = F.resize(image, size)
    return rescaled_image


class RandomResize(object):
    def __init__(self, sizes):
        self.sizes = sizes

    def __call__(self, img, target=None):
        size = random.choice([self.sizes])
        return resize(img, size)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image /= 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image


class Load_data(object):
    def __call__(self, img):
        return torch.from_numpy(img)

def build_transforms(args):
    if args.dataset in ['DVS-GESTURE']:
        scales = 128
        data_transform = {
            "train": transforms.Compose([Load_data(),
                                         RandomHorizontalFlip(),
                                         RandomResize(scales),
                                         Normalize([0.52], [0.15])]),
            "val": transforms.Compose([Load_data(),
                                       RandomResize(scales),
                                       Normalize([0.52], [0.15])])}

    else:
        data_transform = None
        assert args.dataset in ['DVS-GESTURE']
    return data_transform
