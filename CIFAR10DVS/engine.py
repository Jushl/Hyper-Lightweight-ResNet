from spikingjelly.clock_driven import functional
from pathlib import Path
from CIFAR10DVS.util.transforms import build_transforms
import os
import torch
from CIFAR10DVS.util import utils, dataset
import math
import time
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
import math
import random
import numpy as np


def set_seed(_seed_):
    random.seed(_seed_)
    torch.manual_seed(_seed_)
    torch.cuda.manual_seed_all(_seed_)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(_seed_)
    np.random.seed(_seed_)


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, scaler=None, args=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)

        B_size, N_steps, H, W = image.shape
        image = image.view(B_size, -1, N_steps, H, W).permute(2, 0, 1, 3, 4)

        # with torch.autograd.detect_anomaly():
        if scaler is not None:
            with amp.autocast():
                output = model(image.float())
                loss = criterion(output, target)
        else:
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()
            optimizer.step()

        functional.reset_net(model)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        loss_s = loss.item()
        if math.isnan(loss_s):
            raise ValueError('loss is Nan')
        acc1_s = acc1.item()
        acc5_s = acc5.item()

        metric_logger.update(loss=loss_s, lr=optimizer.param_groups[0]["lr"])

        metric_logger.meters['acc1'].update(acc1_s, n=batch_size)
        metric_logger.meters['acc5'].update(acc5_s, n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

        log_dir = args.output_dir
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        file_name = path.parts[-1] + '.out'
        log_out = path / file_name
        learning_rate = optimizer.param_groups[0]["lr"]
        with open(log_out, 'a') as f:
            f.write(f"Epoch:[{epoch}]  loss:[{loss_s}]  lr:[{learning_rate}]  acc1:[{acc1_s}]  acc5:[{acc5_s}]\n")

    metric_logger.synchronize_between_processes()
    return metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg


def evaluate(model, criterion, data_loader, device, print_freq=100, header='Test:', args=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            B_size, N_steps, H, W = image.shape
            image = image.view(B_size, -1, N_steps, H, W).permute(2, 0, 1, 3, 4)

            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image.float())
            loss = criterion(output, target)
            functional.reset_net(model)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    loss, acc1, acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
    print(f' * Acc@1 = {acc1}, Acc@5 = {acc5}, loss = {loss}')

    log_dir = args.output_dir
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    file_name = path.parts[-1] + '.out'
    log_out = path / file_name
    with open(log_out, 'a') as f:
        f.write(f"Val:  acc1 [{acc1}]  acc5 [{acc5}]\n")

    return loss, acc1, acc5


def load_data(traindir, valdir, distributed, args):
    print("Loading data")
    data_transform = build_transforms(args)
    dataset_path = args.data_path
    assert os.path.exists(dataset_path), "{} path does not exist.".format(dataset_path)

    train_dataset = dataset.AedatFolder(root=traindir, transform=data_transform["train"])
    validate_dataset = dataset.AedatFolder(root=valdir, transform=data_transform["val"])
    print("using {} images for training, {} images for validation.".format(len(train_dataset), len(validate_dataset)))

    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(validate_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(validate_dataset)

    return train_dataset, validate_dataset, train_sampler, val_sampler