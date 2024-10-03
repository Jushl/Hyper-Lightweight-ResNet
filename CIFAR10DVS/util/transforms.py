import random
import torch
import numpy as np
import torchvision.transforms.functional as F
from typing import Dict
from torchvision import transforms, datasets

y_mask = 0x7FC00000
y_shift = 22

x_mask = 0x003FF000
x_shift = 12

polarity_mask = 0x800
polarity_shift = 11

valid_mask = 0x80000000
valid_shift = 31

EVT_DVS = 0  # DVS event type
EVT_APS = 1  # APS event


def read_bits(arr, mask=None, shift=None):
    if mask is not None:
        arr = arr & mask
    if shift is not None:
        arr = arr >> shift
    return arr


def skip_header(fp):
    p = 0
    lt = fp.readline()
    ltd = lt.decode().strip()
    while ltd and ltd[0] == "#":
        p += len(lt)
        lt = fp.readline()
        try:
            ltd = lt.decode().strip()
        except UnicodeDecodeError:
            break
    return p


def parse_raw_address(addr,
                      x_mask=x_mask,
                      x_shift=x_shift,
                      y_mask=y_mask,
                      y_shift=y_shift,
                      polarity_mask=polarity_mask,
                      polarity_shift=polarity_shift):
    polarity = read_bits(addr, polarity_mask, polarity_shift).astype(np.bool_)
    x = read_bits(addr, x_mask, x_shift)
    y = read_bits(addr, y_mask, y_shift)
    return x, y, polarity


def load_raw_events(fp, bytes_skip=0, bytes_trim=0, filter_dvs=False, times_first=False):
    p = skip_header(fp)
    fp.seek(p + bytes_skip)
    data = fp.read()
    if bytes_trim > 0:
        data = data[:-bytes_trim]
    data = np.frombuffer(data, dtype='>u4')
    if len(data) % 2 != 0:
        print(data[:20:2])
        print('---')
        print(data[1:21:2])
        raise ValueError('odd number of data elements')
    raw_addr = data[::2]
    timestamp = data[1::2]
    if times_first:
        timestamp, raw_addr = raw_addr, timestamp
    if filter_dvs:
        valid = read_bits(raw_addr, valid_mask, valid_shift) == EVT_DVS
        timestamp = timestamp[valid]
        raw_addr = raw_addr[valid]
    return timestamp, raw_addr


def load_events(fp, filter_dvs=False, **kwargs):
    timestamp, addr = load_raw_events(fp, filter_dvs=filter_dvs,)
    x, y, polarity = parse_raw_address(addr, **kwargs)
    return timestamp, x, y, polarity


def load_origin_data(file_name: str) -> Dict:
    with open(file_name, 'rb') as fp:
        t, x, y, p = load_events(fp,
                                 x_mask=0xfE,
                                 x_shift=1,
                                 y_mask=0x7f00,
                                 y_shift=8,
                                 polarity_mask=1,
                                 polarity_shift=None)
        return {'t': t, 'x': 127 - y, 'y': 127 - x, 'p': 1 - p.astype(int)}


def VoxelGrid(data, TimeSteps):
    data_size = [128, 128]
    voxel_grid = np.zeros((TimeSteps, 128, 128), float).ravel()
    x, y, p, t = data['x'],  data['y'], data['p'], data['t']
    ts = TimeSteps * (t - t[0]) / (t[-1] - t[0])
    p[p == 0] = -1

    tis = ts.astype(int)
    dts = ts - tis
    vals_left = p * (1 - dts)
    vals_right = p * dts

    valid_indices = tis < TimeSteps
    np.add.at(
        voxel_grid,
        tis[valid_indices] * data_size[0] * data_size[1]
        + x[valid_indices]
        + y[valid_indices] * data_size[0],
        vals_left[valid_indices],
    )

    valid_indices = (tis + 1) < TimeSteps
    np.add.at(
        voxel_grid,
        (tis[valid_indices] + 1) * data_size[0] * data_size[1]
        + x[valid_indices]
        + y[valid_indices] * data_size[0],
        vals_right[valid_indices],
    )

    voxel_grid = np.reshape(voxel_grid, (TimeSteps, 1, data_size[1], data_size[0]))

    image = np.zeros((TimeSteps, 128, 128), float)
    for i in range(len(voxel_grid)):
        img = voxel_grid[i][0]

        mean_pos = np.mean(img[img > 0])
        mean_neg = np.mean(img[img < 0])
        img = np.clip(img, a_min=mean_neg * 3, a_max=mean_pos * 3)
        mean_pos = np.mean(img[img > 0])
        mean_neg = np.mean(img[img < 0])
        var_pos = np.var(img[img > 0])
        var_neg = np.var(img[img < 0])
        img = np.clip(img, a_min=mean_neg - 3 * var_neg, a_max=mean_pos + 3 * var_pos)
        max = np.max(img)
        min = np.min(img)
        img[img > 0] /= max
        img[img < 0] /= abs(min)
        map_img = np.zeros_like(img)
        map_img[img < 0] = img[img < 0] * 128 + 128
        map_img[img >= 0] = img[img >= 0] * 127 + 128
        image[i] = map_img

    return image


class ToTensor(object):
    def __init__(self, d=0.1, p=0.5, TS=2):
        self.d = d
        self.p = p
        self.ts = TS

    def __call__(self, event):
        data = load_origin_data(event)
        VoxGrd = VoxelGrid(data, self.ts)
        return torch.from_numpy(VoxGrd)


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


def build_transforms(args):
    if args.dataset in ['CIFAR10DVS']:
        scales = 128
        data_transform = {
            "train": transforms.Compose([ToTensor(TS=args.STEPS),
                                         RandomHorizontalFlip(),
                                         RandomResize(scales),
                                         Normalize([0.403], [0.176])]),
            "val": transforms.Compose([ToTensor(TS=args.STEPS),
                                       RandomResize(scales),
                                       Normalize([0.403], [0.176])])}

    else:
        data_transform = None
        assert args.dataset in ['CIFAR10DVS', 'IMAGENET1K']
    return data_transform
