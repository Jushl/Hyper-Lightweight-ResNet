import os
import torch
import torch.nn as nn
import argparse
import time
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
import datetime
from CIFAR10DVS.util import utils
from CIFAR10DVS.model import build_model
from CIFAR10DVS.engine import train_one_epoch, evaluate, set_seed, load_data


def main(args):
    max_test_acc1 = 0.
    test_acc5_at_max_test_acc1 = 0.
    train_tb_writer = None
    te_tb_writer = None
    utils.init_distributed_mode(args)
    print(args)
    output_dir = args.output_dir
    if args.output_dir:
        utils.mkdir(args.output_dir)
    device = torch.device(args.device)
    train_dir = os.path.join(args.data_path, 'train')
    val_dir = os.path.join(args.data_path, 'val')

    dataset_train, dataset_val, train_sampler, val_sampler = load_data(train_dir, val_dir, args.distributed, args)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size,
        sampler=val_sampler, num_workers=args.workers, pin_memory=True)

    print("Creating model")
    model = build_model(args).to(device)
    criterion = nn.CrossEntropyLoss()

    if args.adamW:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
    elif args.cos_lr:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.cos_lr_T)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        if os.path.exists(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            max_test_acc1 = checkpoint['max_test_acc1']
            test_acc5_at_max_test_acc1 = checkpoint['test_acc5_at_max_test_acc1']

    if args.test_only:
        evaluate(model, criterion, data_loader_val, device=device, header='Test:')
        return

    if args.tb and utils.is_main_process():
        purge_step_train = args.start_epoch
        purge_step_te = args.start_epoch
        train_tb_writer = SummaryWriter(output_dir + '_logs/train', purge_step=purge_step_train)
        te_tb_writer = SummaryWriter(output_dir + '_logs/te', purge_step=purge_step_te)
        with open(output_dir + '_logs/args.txt', 'w', encoding='utf-8') as args_txt:
            args_txt.write(str(args))
        print(f'purge_step_train={purge_step_train}, purge_step_te={purge_step_te}')

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        save_max = False
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_loss, train_acc1, train_acc5 = train_one_epoch(model, criterion, optimizer, data_loader_train, device, epoch,
                                                             args.print_freq, scaler, args)
        if utils.is_main_process():
            train_tb_writer.add_scalar('train_loss', train_loss, epoch)
            train_tb_writer.add_scalar('train_acc1', train_acc1, epoch)
            train_tb_writer.add_scalar('train_acc5', train_acc5, epoch)
        lr_scheduler.step()

        test_loss, test_acc1, test_acc5 = evaluate(model, criterion, data_loader_val, device=device, header='Test:', args=args)
        if te_tb_writer is not None:
            if utils.is_main_process():
                te_tb_writer.add_scalar('test_loss', test_loss, epoch)
                te_tb_writer.add_scalar('test_acc1', test_acc1, epoch)
                te_tb_writer.add_scalar('test_acc5', test_acc5, epoch)

        if max_test_acc1 < test_acc1:
            max_test_acc1 = test_acc1
            test_acc5_at_max_test_acc1 = test_acc5
            save_max = True

        if output_dir:

            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
                'max_test_acc1': max_test_acc1,
                'test_acc5_at_max_test_acc1': test_acc5_at_max_test_acc1,
            }

            utils.save_on_master(
                checkpoint,
                os.path.join(output_dir, 'checkpoint_latest.pth'))

            save_flag = False
            if epoch % 10 == 0:
                save_flag = True

            if save_flag:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(output_dir, f'checkpoint_{epoch}.pth'))

            if save_max:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(output_dir, 'checkpoint_max_test_acc1.pth'))
        print(args)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(output_dir)

        print('Training time {}'.format(total_time_str), 'max_test_acc1', max_test_acc1,
              'test_acc5_at_max_test_acc1', test_acc5_at_max_test_acc1)


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument("--model", default="RESNET18-HE", type=str, metavar="MODEL")
    parser.add_argument('--dataset', default='CIFAR10DVS')
    parser.add_argument("--data_path", default="path-to-data", type=str, help="dataset path")
    parser.add_argument("--num_classes", default=10, type=int, help="number of the classification types")

    parser.add_argument('--STEPS', default=5, type=int)
    parser.add_argument("--channel", default=16)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')

    parser.add_argument('--output-dir', default='', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--lr', default=0.025, type=float, help='initial learning rate')
    parser.add_argument('--lr_drop_list', default=[250, 295])
    parser.add_argument('--multi_step_lr', default=True)
    parser.add_argument('--cos_lr', default=False)
    parser.add_argument('--cos_lr_T', default=80, type=int, help='T_max of CosineAnnealingLR.')

    parser.add_argument('--adamW', default=True, help='The default optimizer is AdamW.')
    parser.add_argument('--wd', '--weight-decay', default=0.0001, dest='weight_decay')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='Momentum for SGD. Adam will not use momentum')

    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--workers', default=16, type=int, metavar='NUM_WORKERS', help='number of data loading workers (default: 16)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--seed', default=42, help='seed')
    parser.add_argument("--cache-dataset", action="store_true",
                        help="Cache the datasets for quicker initialization. It also serializes the transforms",)
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", action="store_true",)
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true",)
    parser.add_argument('--amp', default=True, help='Use AMP training')
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--tb', default=True, help='Use TensorBoard to record logs')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(args)
