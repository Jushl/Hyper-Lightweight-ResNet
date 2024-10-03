from torchvision.datasets import DatasetFolder
from typing import Callable, Dict, Optional, Tuple
from abc import abstractmethod
import struct
import numpy as np
from torchvision.datasets import utils
import torch.utils.data
import os
from concurrent.futures import ThreadPoolExecutor
import time
from torchvision import transforms
import torch
from matplotlib import pyplot as plt
import math
import tqdm
import shutil
from . import configure

np_savez = np.savez_compressed if configure.save_datasets_compressed else np.savez


def play_frame(x: torch.Tensor or np.ndarray, save_gif_to: str = None) -> None:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    to_img = transforms.ToPILImage()
    img_tensor = torch.zeros([x.shape[0], 3, x.shape[2], x.shape[3]])
    img_tensor[:, 1] = x[:, 0]
    img_tensor[:, 2] = x[:, 1]
    if save_gif_to is None:
        while True:
            for t in range(img_tensor.shape[0]):
                    plt.imshow(to_img(img_tensor[t]))
                    plt.pause(0.01)
    else:
        img_list = []
        for t in range(img_tensor.shape[0]):
            img_list.append(to_img(img_tensor[t]))
        img_list[0].save(save_gif_to, save_all=True, append_images=img_list[1:], loop=0)
        print(f'Save frames to [{save_gif_to}].')


def load_aedat_v3(file_name: str) -> Dict:
    with open(file_name, 'rb') as bin_f:
        line = bin_f.readline()
        while line.startswith(b'#'):
            if line == b'#!END-HEADER\r\n':
                break
            else:
                line = bin_f.readline()

        txyp = {
            't': [],
            'x': [],
            'y': [],
            'p': []
        }
        while True:
            header = bin_f.read(28)
            if not header or len(header) == 0:
                break

            # read header
            e_type = struct.unpack('H', header[0:2])[0]
            e_source = struct.unpack('H', header[2:4])[0]
            e_size = struct.unpack('I', header[4:8])[0]
            e_offset = struct.unpack('I', header[8:12])[0]
            e_tsoverflow = struct.unpack('I', header[12:16])[0]
            e_capacity = struct.unpack('I', header[16:20])[0]
            e_number = struct.unpack('I', header[20:24])[0]
            e_valid = struct.unpack('I', header[24:28])[0]

            data_length = e_capacity * e_size
            data = bin_f.read(data_length)
            counter = 0

            if e_type == 1:
                while data[counter:counter + e_size]:
                    aer_data = struct.unpack('I', data[counter:counter + 4])[0]
                    timestamp = struct.unpack('I', data[counter + 4:counter + 8])[0] | e_tsoverflow << 31
                    x = (aer_data >> 17) & 0x00007FFF
                    y = (aer_data >> 2) & 0x00007FFF
                    pol = (aer_data >> 1) & 0x00000001
                    counter = counter + e_size
                    txyp['x'].append(x)
                    txyp['y'].append(y)
                    txyp['t'].append(timestamp)
                    txyp['p'].append(pol)
            else:
                # non-polarity event packet, not implemented
                pass
        txyp['x'] = np.asarray(txyp['x'])
        txyp['y'] = np.asarray(txyp['y'])
        txyp['t'] = np.asarray(txyp['t'])
        txyp['p'] = np.asarray(txyp['p'])
        return txyp


def load_ATIS_bin(file_name: str) -> Dict:
    with open(file_name, 'rb') as bin_f:
        # `& 128` 是取一个8位二进制数的最高位
        # `& 127` 是取其除了最高位，也就是剩下的7位
        raw_data = np.uint32(np.fromfile(bin_f, dtype=np.uint8))
        x = raw_data[0::5]
        y = raw_data[1::5]
        rd_2__5 = raw_data[2::5]
        p = (rd_2__5 & 128) >> 7
        t = ((rd_2__5 & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
    return {'t': t, 'x': x, 'y': y, 'p': p}


def load_npz_frames(file_name: str) -> np.ndarray:
    return np.load(file_name, allow_pickle=True)['frames'].astype(np.float32)


def integrate_events_segment_to_frame(x: np.ndarray, y: np.ndarray, p: np.ndarray, H: int, W: int, j_l: int = 0, j_r: int = -1) -> np.ndarray:
    frame = np.zeros(shape=[2, H * W])
    x = x[j_l: j_r].astype(int)  # avoid overflow
    y = y[j_l: j_r].astype(int)
    p = p[j_l: j_r]
    mask = []
    mask.append(p == 0)
    mask.append(np.logical_not(mask[0]))
    for c in range(2):
        position = y[mask[c]] * W + x[mask[c]]
        events_number_per_pos = np.bincount(position)
        frame[c][np.arange(events_number_per_pos.size)] += events_number_per_pos
    return frame.reshape((2, H, W))


def cal_fixed_frames_number_segment_index(events_t: np.ndarray, split_by: str, frames_num: int) -> tuple:
    j_l = np.zeros(shape=[frames_num], dtype=int)
    j_r = np.zeros(shape=[frames_num], dtype=int)
    N = events_t.size

    if split_by == 'number':
        di = N // frames_num
        for i in range(frames_num):
            j_l[i] = i * di
            j_r[i] = j_l[i] + di
        j_r[-1] = N

    elif split_by == 'time':
        dt = (events_t[-1] - events_t[0]) // frames_num
        idx = np.arange(N)
        for i in range(frames_num):
            t_l = dt * i + events_t[0]
            t_r = t_l + dt
            mask = np.logical_and(events_t >= t_l, events_t < t_r)
            idx_masked = idx[mask]
            j_l[i] = idx_masked[0]
            j_r[i] = idx_masked[-1] + 1

        j_r[-1] = N
    else:
        raise NotImplementedError

    return j_l, j_r


def VoxelGrid(data, TimeSteps, H, W):
    data_size = [H, W]
    voxel_grid = np.zeros((TimeSteps, H, W), float).ravel()
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

    image = np.zeros((TimeSteps, H, W), float)
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


def integrate_events_by_fixed_frames_number(events: Dict, split_by: str, frames_num: int, H: int, W: int) -> np.ndarray:
    t, x, y, p = (events[key] for key in ('t', 'x', 'y', 'p'))
    j_l, j_r = cal_fixed_frames_number_segment_index(t, split_by, frames_num)
    frames = np.zeros([frames_num, 2, H, W])
    for i in range(frames_num):
        frames[i] = integrate_events_segment_to_frame(x, y, p, H, W, j_l[i], j_r[i])
    return frames  # frames是一个[5, 2, 128, 128]的ndarray


def integrate_events_to_VoxelGrid_by_fixed_frames_number(events: Dict, split_by: str, frames_num: int, H: int, W: int) -> np.ndarray:
    return VoxelGrid(events, frames_num, H, W)


def integrate_events_file_to_frames_file_by_fixed_frames_number(loader: Callable, events_np_file: str, output_dir: str, split_by: str, frames_num: int, H: int, W: int, print_save: bool = False) -> None:
    fname = os.path.join(output_dir, os.path.basename(events_np_file))
    # np_savez(fname, frames=integrate_events_by_fixed_frames_number(loader(events_np_file), split_by, frames_num, H, W))
    np_savez(fname, frames=integrate_events_to_VoxelGrid_by_fixed_frames_number(loader(events_np_file), split_by, frames_num, H, W))
    if print_save:
        print(f'Frames [{fname}] saved.')


def integrate_events_by_fixed_duration(events: Dict, duration: int, H: int, W: int) -> np.ndarray:
    x = events['x']
    y = events['y']
    t = events['t']
    p = events['p']
    N = t.size

    frames = []
    left = 0
    right = 0
    while True:
        t_l = t[left]
        while True:
            if right == N or t[right] - t_l > duration:
                break
            else:
                right += 1
        # integrate from index [left, right)
        frames.append(np.expand_dims(integrate_events_segment_to_frame(x, y, p, H, W, left, right), 0))

        left = right

        if right == N:
            return np.concatenate(frames)


def integrate_events_file_to_frames_file_by_fixed_duration(loader: Callable, events_np_file: str, output_dir: str, duration: int, H: int, W: int, print_save: bool = False) -> None:
    frames = integrate_events_by_fixed_duration(loader(events_np_file), duration, H, W)
    fname, _ = os.path.splitext(os.path.basename(events_np_file))
    fname = os.path.join(output_dir, f'{fname}_{frames.shape[0]}.npz')
    np_savez(fname, frames=frames)
    if print_save:
        print(f'Frames [{fname}] saved.')
    return frames.shape[0]


def save_frames_to_npz_and_print(fname: str, frames):
    np_savez(fname, frames=frames)
    print(f'Frames [{fname}] saved.')

def create_same_directory_structure(source_dir: str, target_dir: str) -> None:
    for sub_dir_name in os.listdir(source_dir):
        source_sub_dir = os.path.join(source_dir, sub_dir_name)
        if os.path.isdir(source_sub_dir):
            target_sub_dir = os.path.join(target_dir, sub_dir_name)
            os.mkdir(target_sub_dir)
            print(f'Mkdir [{target_sub_dir}].')
            create_same_directory_structure(source_sub_dir, target_sub_dir)


def split_to_train_test_set(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int, random_split: bool = False):
    label_idx = []
    for i in range(num_classes):
        label_idx.append([])

    for i, item in enumerate(tqdm.tqdm(origin_dataset)):
        y = item[1]
        if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
            y = y.item()
        label_idx[y].append(i)
    train_idx = []
    test_idx = []
    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])

    for i in range(num_classes):
        pos = math.ceil(label_idx[i].__len__() * train_ratio)
        train_idx.extend(label_idx[i][0: pos])
        test_idx.extend(label_idx[i][pos: label_idx[i].__len__()])

    return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)


def pad_sequence_collate(batch: list):
    x_list = []
    x_len_list = []
    y_list = []
    for x, y in batch:
        x_list.append(torch.as_tensor(x))
        x_len_list.append(x.shape[0])
        y_list.append(y)

    return torch.nn.utils.rnn.pad_sequence(x_list, batch_first=True), torch.as_tensor(y_list), torch.as_tensor(x_len_list)


class NeuromorphicDatasetFolder(DatasetFolder):
    def __init__(
            self,
            root: str,
            train: bool = None,
            data_type: str = 'event',
            frames_number: int = None,
            split_by: str = None,
            duration: int = None,
            custom_integrate_function: Callable = None,
            custom_integrated_frames_dir_name: str = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        events_np_root = os.path.join(root, 'events_np')
        if not os.path.exists(events_np_root):
            download_root = os.path.join(root, 'download')
            if os.path.exists(download_root):
                print(f'The [{download_root}] directory for saving downloaded files already exists, check files...')
                resource_list = self.resource_url_md5()
                for i in range(resource_list.__len__()):
                    file_name, url, md5 = resource_list[i]
                    fpath = os.path.join(download_root, file_name)
                    if not utils.check_integrity(fpath=fpath, md5=md5):
                        print(f'The file [{fpath}] does not exist or is corrupted.')
                        if os.path.exists(fpath):
                            os.remove(fpath)
                            print(f'Remove [{fpath}]')
                        if self.downloadable():
                            print(f'Download [{file_name}] from [{url}] to [{download_root}]')
                            utils.download_url(url=url, root=download_root, filename=file_name, md5=md5)
                        else:
                            raise NotImplementedError(
                                f'This dataset can not be downloaded by SpikingJelly, please download [{file_name}] from [{url}] manually and put files at {download_root}.')
            else:
                os.mkdir(download_root)
                print(f'Mkdir [{download_root}] to save downloaded files.')
                resource_list = self.resource_url_md5()
                if self.downloadable():
                    for i in range(resource_list.__len__()):
                        file_name, url, md5 = resource_list[i]
                        print(f'Download [{file_name}] from [{url}] to [{download_root}]')
                        utils.download_url(url=url, root=download_root, filename=file_name, md5=md5)
                else:
                    raise NotImplementedError(f'This dataset can not be downloaded by SpikingJelly, '
                                              f'please download files manually and put files at [{download_root}]. '
                                              f'The resources file_name, url, and md5 are: \n{resource_list}')
            extract_root = os.path.join(root, 'extract')
            if os.path.exists(extract_root):
                print(f'The directory [{extract_root}] for saving extracted files already exists.\n'
                      f'SpikingJelly will not check the data integrity of extracted files.\n'
                      f'If extracted files are not integrated, please delete [{extract_root}] manually, '
                      f'then SpikingJelly will re-extract files from [{download_root}].')
            else:
                os.mkdir(extract_root)
                print(f'Mkdir [{extract_root}].')
                self.extract_downloaded_files(download_root, extract_root)
            os.mkdir(events_np_root)
            print(f'Mkdir [{events_np_root}].')
            print(f'Start to convert the origin data from [{extract_root}] to [{events_np_root}] in np.ndarray format.')
            self.create_events_np_files(extract_root, events_np_root)

        H, W = self.get_H_W()

        if data_type == 'event':
            _root = events_np_root
            _loader = np.load
            _transform = transform
            _target_transform = target_transform

        elif data_type == 'frame':
            if frames_number is not None:
                assert frames_number > 0 and isinstance(frames_number, int)
                assert split_by == 'time' or split_by == 'number'
                frames_np_root = os.path.join(root, f'frames_number_{frames_number}_split_by_{split_by}')
                if os.path.exists(frames_np_root):
                    print(f'The directory [{frames_np_root}] already exists.')
                else:
                    os.mkdir(frames_np_root)
                    print(f'Mkdir [{frames_np_root}].')

                    # create the same directory structure
                    create_same_directory_structure(events_np_root, frames_np_root)

                    # use multi-thread to accelerate
                    t_ckp = time.time()
                    with ThreadPoolExecutor(max_workers=configure.max_threads_number_for_datasets_preprocess) as tpe:
                        print(f'Start ThreadPoolExecutor with max workers = [{tpe._max_workers}].')
                        for e_root, e_dirs, e_files in os.walk(events_np_root):
                            if e_files.__len__() > 0:
                                output_dir = os.path.join(frames_np_root, os.path.relpath(e_root, events_np_root))
                                for e_file in e_files:  # 开始处理npz文件
                                    events_np_file = os.path.join(e_root, e_file)
                                    print(f'Start to integrate [{events_np_file}] to frames and save to [{output_dir}].')
                                    tpe.submit(integrate_events_file_to_frames_file_by_fixed_frames_number, self.load_events_np, events_np_file, output_dir, split_by, frames_number, H, W, True)

                    print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')

                _root = frames_np_root
                _loader = load_npz_frames
                _transform = transform
                _target_transform = target_transform

            elif duration is not None:
                assert duration > 0 and isinstance(duration, int)
                frames_np_root = os.path.join(root, f'duration_{duration}')
                if os.path.exists(frames_np_root):
                    print(f'The directory [{frames_np_root}] already exists.')

                else:
                    os.mkdir(frames_np_root)
                    print(f'Mkdir [{frames_np_root}].')
                    # create the same directory structure
                    create_same_directory_structure(events_np_root, frames_np_root)
                    # use multi-thread to accelerate
                    t_ckp = time.time()
                    with ThreadPoolExecutor(max_workers=configure.max_threads_number_for_datasets_preprocess) as tpe:
                        print(f'Start ThreadPoolExecutor with max workers = [{tpe._max_workers}].')
                        for e_root, e_dirs, e_files in os.walk(events_np_root):
                            if e_files.__len__() > 0:
                                output_dir = os.path.join(frames_np_root, os.path.relpath(e_root, events_np_root))
                                for e_file in e_files:
                                    events_np_file = os.path.join(e_root, e_file)
                                    print(f'Start to integrate [{events_np_file}] to frames and save to [{output_dir}].')
                                    tpe.submit(integrate_events_file_to_frames_file_by_fixed_duration, self.load_events_np, events_np_file, output_dir, duration, H, W, True)

                    print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')

                _root = frames_np_root
                _loader = load_npz_frames
                _transform = transform
                _target_transform = target_transform

            elif custom_integrate_function is not None:
                if custom_integrated_frames_dir_name is None:
                    custom_integrated_frames_dir_name = custom_integrate_function.__name__

                frames_np_root = os.path.join(root, custom_integrated_frames_dir_name)
                if os.path.exists(frames_np_root):
                    print(f'The directory [{frames_np_root}] already exists.')
                else:
                    os.mkdir(frames_np_root)
                    print(f'Mkdir [{frames_np_root}].')
                    # create the same directory structure
                    create_same_directory_structure(events_np_root, frames_np_root)
                    # use multi-thread to accelerate
                    t_ckp = time.time()
                    with ThreadPoolExecutor(max_workers=configure.max_threads_number_for_datasets_preprocess) as tpe:
                        print(f'Start ThreadPoolExecutor with max workers = [{tpe._max_workers}].')
                        for e_root, e_dirs, e_files in os.walk(events_np_root):
                            if e_files.__len__() > 0:
                                output_dir = os.path.join(frames_np_root, os.path.relpath(e_root, events_np_root))
                                for e_file in e_files:
                                    events_np_file = os.path.join(e_root, e_file)
                                    print(
                                        f'Start to integrate [{events_np_file}] to frames and save to [{output_dir}].')
                                    tpe.submit(save_frames_to_npz_and_print, os.path.join(output_dir, os.path.basename(events_np_file)), custom_integrate_function(np.load(events_np_file), H, W))

                    print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')

                _root = frames_np_root
                _loader = load_npz_frames
                _transform = transform
                _target_transform = target_transform

            else:
                raise ValueError('At least one of "frames_number", "duration" and "custom_integrate_function" should not be None.')

        if train is not None:
            if train:
                _root = os.path.join(_root, 'train')
            else:
                _root = os.path.join(_root, 'test')
        else:
            _root = self.set_root_when_train_is_none(_root)

        super().__init__(root=_root, loader=_loader, extensions=('.npz', ), transform=_transform,
                         target_transform=_target_transform)

    def set_root_when_train_is_none(self, _root: str):
        return _root


    @staticmethod
    @abstractmethod
    def resource_url_md5() -> list:
        pass

    @staticmethod
    @abstractmethod
    def downloadable() -> bool:
        pass

    @staticmethod
    @abstractmethod
    def extract_downloaded_files(download_root: str, extract_root: str):
        pass

    @staticmethod
    @abstractmethod
    def create_events_np_files(extract_root: str, events_np_root: str):
        pass

    @staticmethod
    @abstractmethod
    def get_H_W() -> Tuple:
        pass

    @staticmethod
    def load_events_np(fname: str):
        return np.load(fname)



def random_temporal_delete(x_seq: torch.Tensor or np.ndarray, T_remain: int, batch_first):
    """
    :param x_seq: a sequence with `shape = [T, N, *]`, where `T` is the sequence length and `N` is the batch size
    :type x_seq: torch.Tensor or np.ndarray
    :param T_remain: the remained length
    :type T_remain: int
    :param batch_first: if `True`, `x_seq` will be regarded as `shape = [N, T, *]`
    :type batch_first: bool
    :return: the sequence with length `T_remain`, which is obtained by randomly removing `T - T_remain` slices
    :rtype: torch.Tensor or np.ndarray
    The random temporal delete data augmentation used in `Deep Residual Learning in Spiking Neural Networks <https://arxiv.org/abs/2102.04159>`_.
    Codes example:

    .. code-block:: python

        import torch
        from spikingjelly.datasets import random_temporal_delete
        T = 8
        T_remain = 5
        N = 4
        x_seq = torch.arange(0, N*T).view([N, T])
        print('x_seq=\\n', x_seq)
        print('random_temporal_delete(x_seq)=\\n', random_temporal_delete(x_seq, T_remain, batch_first=True))

    Outputs:

    .. code-block:: shell

        x_seq=
         tensor([[ 0,  1,  2,  3,  4,  5,  6,  7],
                [ 8,  9, 10, 11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20, 21, 22, 23],
                [24, 25, 26, 27, 28, 29, 30, 31]])
        random_temporal_delete(x_seq)=
         tensor([[ 0,  1,  4,  6,  7],
                [ 8,  9, 12, 14, 15],
                [16, 17, 20, 22, 23],
                [24, 25, 28, 30, 31]])
    """
    if batch_first:
        sec_list = np.random.choice(x_seq.shape[1], T_remain, replace=False)
    else:
        sec_list = np.random.choice(x_seq.shape[0], T_remain, replace=False)
    sec_list.sort()
    if batch_first:
        return x_seq[:, sec_list]
    else:
        return x_seq[sec_list]

class RandomTemporalDelete(torch.nn.Module):
    def __init__(self, T_remain: int, batch_first: bool):
        """
        :param T_remain: the remained length
        :type T_remain: int
        :type T_remain: int
        :param batch_first: if `True`, `x_seq` will be regarded as `shape = [N, T, *]`
        The random temporal delete data augmentation used in `Deep Residual Learning in Spiking Neural Networks <https://arxiv.org/abs/2102.04159>`_.
        Refer to :class:`random_temporal_delete` for more details.
        """
        super().__init__()
        self.T_remain = T_remain
        self.batch_first = batch_first

    def forward(self, x_seq: torch.Tensor or np.ndarray):
        return random_temporal_delete(x_seq, self.T_remain, self.batch_first)


def create_sub_dataset(source_dir: str, target_dir:str, ratio: float, use_soft_link=True, randomly=False):
    """
    :param source_dir: the directory path of the origin dataset
    :type source_dir: str
    :param target_dir: the directory path of the sub dataset
    :type target_dir: str
    :param ratio: the ratio of samples sub dataset will copy from the origin dataset
    :type ratio: float
    :param use_soft_link: if ``True``, the sub dataset will use soft link to copy; else, the sub dataset will copy files
    :type use_soft_link: bool
    :param randomly: if ``True``, the files copy from the origin dataset will be picked up randomly. The randomness is controlled by
            ``numpy.random.seed``
    :type randomly: bool
    Create a sub dataset with copy ``ratio`` of samples from the origin dataset.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f'Mkdir [{target_dir}].')
    create_same_directory_structure(source_dir, target_dir)
    warnings_info = []
    for e_root, e_dirs, e_files in os.walk(source_dir, followlinks=True):
        if e_files.__len__() > 0:
            output_dir = os.path.join(target_dir, os.path.relpath(e_root, source_dir))
            if ratio >= 1.:
                samples_number = e_files.__len__()
            else:
                samples_number = int(ratio * e_files.__len__())
            if samples_number == 0:
                warnings_info.append(f'Warning: the samples number is 0 in [{output_dir}].')
            if randomly:
                np.random.shuffle(e_files)
            for i, e_file in enumerate(e_files):
                if i >= samples_number:
                    break
                source_file = os.path.join(e_root, e_file)
                target_file = os.path.join(output_dir, os.path.basename(source_file))
                if use_soft_link:
                    os.symlink(source_file, target_file)
                    # print(f'symlink {source_file} -> {target_file}')
                else:
                    shutil.copyfile(source_file, target_file)
                    # print(f'copyfile {source_file} -> {target_file}')
            print(f'[{samples_number}] files in [{e_root}] have been copied to [{output_dir}].')

    for i in range(warnings_info.__len__()):
        print(warnings_info[i])


class DVS128Gesture(NeuromorphicDatasetFolder):
    def __init__(
            self,
            root: str,
            train: bool = None,
            data_type: str = 'event',
            frames_number: int = None,
            split_by: str = None,
            duration: int = None,
            custom_integrate_function: Callable = None,
            custom_integrated_frames_dir_name: str = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:

        assert train is not None
        super().__init__(root, train, data_type, frames_number, split_by, duration, custom_integrate_function,
                         custom_integrated_frames_dir_name, transform, target_transform)

    @staticmethod
    def resource_url_md5() -> list:

        url = 'https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794'
        return [
            ('DvsGesture.tar.gz', url, '8a5c71fb11e24e5ca5b11866ca6c00a1'),
            ('gesture_mapping.csv', url, '109b2ae64a0e1f3ef535b18ad7367fd1'),
            ('LICENSE.txt', url, '065e10099753156f18f51941e6e44b66'),
            ('README.txt', url, 'a0663d3b1d8307c329a43d949ee32d19')
        ]

    @staticmethod
    def downloadable() -> bool:
        return False


    @staticmethod
    def split_aedat_files_to_np(fname: str, aedat_file: str, csv_file: str, output_dir: str):
        events = DVS128Gesture.load_origin_data(aedat_file)
        print(f'Start to split [{aedat_file}] to samples.')
        csv_data = np.loadtxt(csv_file, dtype=np.uint32, delimiter=',', skiprows=1)
        label_file_num = [0] * 11
        for i in range(csv_data.shape[0]):
            label = csv_data[i][0] - 1
            t_start = csv_data[i][1]
            t_end = csv_data[i][2]
            mask = np.logical_and(events['t'] >= t_start, events['t'] < t_end)
            file_name = os.path.join(output_dir, str(label), f'{fname}_{label_file_num[label]}.npz')
            np_savez(file_name,
                     t=events['t'][mask],
                     x=events['x'][mask],
                     y=events['y'][mask],
                     p=events['p'][mask]
                     )
            print(f'[{file_name}] saved.')
            label_file_num[label] += 1



    @staticmethod
    def get_H_W() -> Tuple:
        return 128, 128
