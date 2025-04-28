import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
import os
import gc
import math
import warnings
import builtins

warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.parallel
# from torch.nn import init
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import Subset, DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from dataset import DistributedRandomSampler
from lib.augmentation import MinMaxNorm
from utils import parse_device, BlankSummaryWriter
from torch.utils.tensorboard import SummaryWriter
from conf import settings
from progress.bar import Bar as Bar
from quant import prepare, QuantLinear, find_beta, find_the, find_beta_with_name, find_the_with_name, find_error
# from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup
from warmup_scheduler import GradualWarmupScheduler
from models.resnet_pytorch_imagenet_simple import resnet50, resnet18

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default=r'/data/ILSVRC2012',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument("--device_id", action="store", default="",
                    help="device_id, eg: 0 for single #0 gpu or 0,1,2 for multi-gpus (#0, #1 and #2). Set any negtive num (eg: -1) if only CPU is available")
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--rank', default=-1, type=int,
                    help='global rank for distributed training (Do not touch, deprecated)')
parser.add_argument("--local-rank", default=-1, type=int, help="local rank for distributed training (Do not touch, deprecated)")
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training (Do not touch)')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training (Do not touch)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")

parser.add_argument('--debug_mode', action='store_true', help="no writer when running")

best_acc1 = 0
NUM_CLASSES = 1000

counter_for_recording_beta = 0
counter_for_recording_theta = 0
counter_for_recording_lr = 0


def main(args):
    global best_acc1
    device_list = parse_device(args.device_id)

    rank = args.rank
    local_rank = args.local_rank
    global_rank = rank
    # global_rank = torch.distributed.get_rank()

    if device_list != "cpu":
        use_gpu = True
        cudnn.benchmark = True
        if args.distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            ngpus_per_node = torch.cuda.device_count()
            if ngpus_per_node == 1 and args.dist_backend == "nccl":
                warnings.warn(
                    "nccl backend >=2.5 requires GPU count>1, see https://github.com/NVIDIA/nccl/issues/103 perhaps use 'gloo'")
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        else:
            device = torch.device("cuda:" + str(device_list[0]))
    else:
        use_gpu = False
        device = torch.device("cpu")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    checkpoint_folder = os.path.join(r"./checkpoint_imagenet", args.arch, settings.TIME_NOW)
    # create checkpoint folder to save model
    if rank == 0:
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder, exist_ok=True)

        # use tensorboard
        if not os.path.exists(settings.LOG_DIR):
            os.makedirs(settings.LOG_DIR, exist_ok=True)

    # since tensorboard can't overwrite old values
    # so the only way is to create a new tensorboard log
    if args.evaluate:
        writer = BlankSummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.arch, settings.TIME_NOW))
    elif args.distributed:
        dist.barrier()
        if rank == 0:
            writer = SummaryWriter(log_dir=os.path.join(
                settings.LOG_DIR, args.arch, settings.TIME_NOW))
        else:
            writer = BlankSummaryWriter(log_dir=os.path.join(
                settings.LOG_DIR, args.arch, settings.TIME_NOW))
    else:
        writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.arch, settings.TIME_NOW))

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        if args.arch == "resnet18":
            model = models.__dict__[args.arch](pretrained=True, num_classes=NUM_CLASSES)
        elif args.arch == "resnet50":
            model = models.__dict__[args.arch](pretrained=True, num_classes=NUM_CLASSES)
        else:
            model = models.__dict__[args.arch](pretrained=True, num_classes=NUM_CLASSES)
    else:
        print("=> creating model '{}'".format(args.arch))
        # model = models.__dict__[args.arch](num_classes=NUM_CLASSES)
        if args.arch == "resnet18":
            model = resnet18(num_classes=NUM_CLASSES)
            pretrained_path = r""
            checkpoint = torch.load(pretrained_path, map_location="cpu")['state_dict']
            new_state_dict = {}
            for k, v in checkpoint.items():
                new_state_dict[k[7:]] = v
            model.load_state_dict(new_state_dict)
        elif args.arch == "resnet50":
            model = resnet50(num_classes=NUM_CLASSES)
            pretrained_path = r""
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            model.load_state_dict(checkpoint)
        else:
            model = models.__dict__[args.arch](num_classes=NUM_CLASSES)

    # Data loading code
    if args.dummy:
        print("=> Dummy data is used!")
        train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
    else:
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                # transforms.RandomRotation(45),
                # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                transforms.ToTensor(),
                normalize,
                # MinMaxNorm(new_min=0, new_max=1)
            ]))

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
                # MinMaxNorm(new_min=0, new_max=1)
            ]))

    if args.distributed:
        train_sampler = DistributedRandomSampler(train_dataset, num_samples=len(train_dataset) // 8)  # only for distributed
        val_sampler = DistributedSampler(val_dataset)
    else:
        train_sampler = RandomSampler(train_dataset, num_samples=len(train_dataset) // 8)
        val_sampler = None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler)

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size // 2, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=val_sampler)

    # for name, module in model.named_modules():
    #     if isinstance(module, nn.Linear) and hasattr(module, 'bias'):
    #         delattr(module, 'bias')
    #         module.register_parameter('bias', None)

    # init_params(model)

    # print(f"original model: {model}")

    dummy_input = val_loader.__iter__().__next__()[0]
    print(dummy_input.shape)
    model = prepare(model, inplace=False, a_bits=8, w_bits=8, fuse_model=False, quant_inference=False,
                    calibration_input=dummy_input, writer=writer)
    # print(f"quantized model: {model}")
    # input_tensor = torch.Tensor(1, 3, 224, 224)
    # if args.gpu:
    #     input_tensor = input_tensor.cuda()
    # writer.add_graph(model, input_tensor)

    beta_paras = find_beta(model)
    print(f"find {len(beta_paras)} beta paras")
    the_paras = find_the(model)
    print(f"find {len(the_paras)} theta paras")
    other_paras = list(set(model.parameters()) - set(beta_paras) - set(the_paras))
    # param_lr_set = [{'params': beta_paras, 'lr': 0.1},
    #                 {'params': the_paras, 'lr': 0.0001},
    #                 {'params': other_paras, 'lr': args.lr}]
    if args.resume:
        # for resnet18, set all to 0.1, for resnet50 set beta and theta to 0.01, others to 0.1
        param_lr_set = [{'params': beta_paras, 'lr': 0.001, 'weight_decay': 0},
                        {'params': the_paras, 'lr': 0.001, 'weight_decay': 0},
                        {'params': other_paras, 'lr': args.lr, 'weight_decay': 0}]
    else:
        param_lr_set = [{'params': beta_paras, 'lr': args.lr, 'weight_decay': args.weight_decay},
                        {'params': the_paras, 'lr': args.lr, 'weight_decay': args.weight_decay},
                        {'params': other_paras, 'lr': args.lr, 'weight_decay': args.weight_decay}]

    if use_gpu:
        model.to(device)
        if len(device_list) > 1:
            print(f"DataParallel will divide and allocate batch_size to all available GPUs: {device_list}")
            if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
                if args.distributed:
                    model.features = torch.nn.parallel.DistributedDataParallel(model.features)
                else:
                    model.features = torch.nn.DataParallel(model.features, device_ids=device_list)
            else:
                if args.distributed:
                    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
                else:
                    model = torch.nn.DataParallel(model, device_ids=device_list)

            # model.cuda(device_list[0])
        else:
            # torch.cuda.set_device(device)
            # model = model.cuda(device_list[0])
            model = torch.nn.DataParallel(model, device_ids=device_list).cuda()
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)  # criterion = nn.CrossEntropyLoss().cuda()
    # criterion = focal_loss(num_classes=1000).to(device)
    # optimizer = optim.Adam(param_lr_set, lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(param_lr_set, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = optim.Adadelta(param_lr_set, lr=args.lr, weight_decay=0.0)  # for quant version

    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    # scheduler = CosineAnnealingLR(optimizer, T_max=120, eta_min=0.00001)

    # lr_scheduler_warmup = create_lr_scheduler_with_warmup(scheduler,
    #                                                       warmup_start_value=0.001,
    #                                                       warmup_duration=3,
    #                                                       warmup_end_value=args.lr)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=5, warm_up_init_lr=0.001,
                                                 after_scheduler=scheduler, expo=True)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if use_gpu:
                # Map model to be loaded to specified single gpu.
                checkpoint = torch.load(args.resume, map_location=device)
            else:
                checkpoint = torch.load(args.resume)
            # args.start_epoch = checkpoint['epoch']
            # best_acc1 = checkpoint['best_acc1']
            # if use_gpu:
            #     # best_acc1 may be from a checkpoint from a different GPU
            #     best_acc1 = best_acc1.to(device)
            global counter_for_recording_beta, counter_for_recording_theta, counter_for_recording_lr
            counter_for_recording_beta = checkpoint.get('counter_for_recording_beta', 0)
            counter_for_recording_theta = checkpoint.get('counter_for_recording_theta', 0)
            counter_for_recording_lr = checkpoint.get('counter_for_recording_lr', 0)
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
        resume_epoch = 43
    else:
        resume_epoch = -1

    if args.evaluate:
        validate(val_loader, model, criterion, 0, args, use_gpu, device)
        return

    # for m in model.modules():

    for m in model.modules():
        # # theta of last fc layer is fixed
        # if hasattr(m, "weight_quantizer"):
        #     m.weight_quantizer.the.freeze = True
        if hasattr(m, "recording"):
            m.rank = rank

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        current_lr = []
        for group in optimizer.param_groups:
            current_lr.append(group["lr"])
        print(f"\n[{epoch}] current lr: {current_lr}")

        if args.distributed:
            dist.barrier()
        freeze_if_necessary(model, epoch, args)
        if args.distributed:
            dist.barrier()
        if args.resume:
            if epoch < resume_epoch:
                # scheduler.step()
                scheduler.step()
                print(f"skip epoch {epoch}")
                continue

        # train for one epoch
        if epoch < 5:

            param_lr_set = [{'params': beta_paras, 'lr': 0.0, 'weight_decay': 0.0},
                            {'params': the_paras, 'lr': 0.0, 'weight_decay': 0.0},
                            {'params': other_paras, 'lr': 0.001, 'weight_decay': 1e-5}]
            temp_optimizer = torch.optim.SGD(param_lr_set, 0.001,
                            momentum=args.momentum,
                            weight_decay=1e-5)
            train_losses, train_top1, train_top5 = train(train_loader, model, criterion, temp_optimizer, epoch, device, args, writer)
        else:
            train_losses, train_top1, train_top5 = train(train_loader, model, criterion, optimizer, epoch, device, args, writer)
        # evaluate on validation set
        acc1, top5 = validate(val_loader, model, criterion, epoch, args, use_gpu, device)

        if rank == 0:
            # update training loss for each iteration
            writer.add_scalar("Loss/train", train_losses, epoch)
            writer.add_scalar("Acc1/train", train_top1, epoch)
            writer.add_scalar("Acc5/train", train_top5, epoch)
            writer.add_scalar("Acc1/val", acc1, epoch)
            writer.add_scalar("Acc5/val", top5, epoch)

        # adjust_learning_rate(optimizer, epoch)
        # scheduler.step()
        # lr_scheduler_warmup(None)
        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if local_rank == 0:
            print(f"best acc={best_acc1}, current acc={acc1}")

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, is_best, "checkpoint.pth", checkpoint_folder)
            # print("=> saving checkpoint")
        writer.close()


def freeze_if_necessary(model, epoch, args):
    frozen_epoch = [[30, 40], [60, 70], [90, args.epochs]]

    if epoch in [row[0] for row in frozen_epoch]:
        for m in model.modules():
            if hasattr(m, "activation_quantizer"):
                m.activation_quantizer.beta.freeze = True
        if args.local_rank == 0:
            print("betas have been frozen!")
    elif epoch == args.start_epoch or epoch in [row[1] for row in frozen_epoch]:
        for m in model.modules():
            if hasattr(m, "activation_quantizer"):
                m.activation_quantizer.beta.freeze = False
        if args.local_rank == 0:
            print("betas have been unfrozen!")


def train(train_loader, model, criterion, optimizer, epoch, device, args, writer=None):
    global counter_for_recording_beta, counter_for_recording_theta, counter_for_recording_lr
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    data_time = AverageMeter('Data', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.AVERAGE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    bar = Bar(f'Training [{epoch}/{args.epochs}]', max=len(train_loader))
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # move data to the same device as model
        images = images.cuda()
        target = target.cuda()

        # compute output
        output = model(images)
        # output = torch.randn(images.size(0), 1000).to(device)
        loss = criterion(output, target)
        error_loss = torch.mean(torch.stack(find_error(model)))
        loss = loss + error_loss

        # beta_reg = torch.mean(torch.stack(find_beta(model)) ** 2)
        # loss = loss + 0.0001 * beta_reg
        # beta_loss = torch.mean(torch.stack(beta_loss))
        # loss = loss + beta_loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        # for m in model.modules():
        #     if hasattr(m, "activation_quantizer"):
        #         b_grad = m.activation_quantizer.beta.grad
        #         if b_grad is None:
        #             continue
        #         b_error = m.activation_quantizer.beta_error
        #
        #         # b_grad.data += 0.001 * m.activation_quantizer.beta_error
        #         b_grad.data += torch.sign(b_grad) * (b_error.abs())
        optimizer.step()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target.data, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.rank == 0:
            if i == 0:
                for m in model.modules():
                    if hasattr(m, "recording"):
                        m.recording = True
                        m.record_epoch = epoch
            elif i == 1:
                for m in model.modules():
                    if hasattr(m, "recording"):
                        m.recording = False

            if i % 100 == 0:
                beta_paras, names = find_beta_with_name(model)
                lr_sample = optimizer.param_groups[0]['lr']
                writer.add_scalar('lr_beta', lr_sample, counter_for_recording_lr)
                lr_sample = optimizer.param_groups[1]['lr']
                writer.add_scalar('lr_theta', lr_sample, counter_for_recording_lr)
                lr_sample = optimizer.param_groups[2]['lr']
                writer.add_scalar('lr_other', lr_sample, counter_for_recording_lr)
                counter_for_recording_lr += 1
                if len(beta_paras) > 0:
                    tag_scalar_dict = {name: beta for name, beta in zip(names, beta_paras)}
                    writer.add_scalars('beta_values', tag_scalar_dict, counter_for_recording_beta)
                    counter_for_recording_beta += 1
                theta_paras, names = find_the_with_name(model)
                if len(theta_paras) > 0:
                    tag_scalar_dict = {name: theta for name, theta in zip(names, theta_paras)}
                    writer.add_scalars('theta_values', tag_scalar_dict, counter_for_recording_theta)
                    counter_for_recording_theta += 1

            # if i % args.print_freq == 0:
            #     progress.display(i + 1)
            # plot progress

        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=i + 1,
            size=len(train_loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
    bar.finish()
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()
    return losses.avg, top1.avg, top5.avg
    # return 0.,0.,0.


def validate(val_loader, model, criterion, epoch, args, use_gpu=True, device=None, writer=None):
    data_time = AverageMeter('Data', ':6.3f', Summary.NONE)
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    # for distributed is "len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)))"
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        if args.local_rank == 0:
            bar = Bar(f'Validating [{epoch}/{args.epochs}]', max=len(val_loader))
        for i, (images, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            if use_gpu:
                images = images.cuda()
                target = target.cuda()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output.data, target.data, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            #     progress.display(i + 1)
            # plot progress
            if args.local_rank == 0:
                bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=i + 1,
                    size=len(val_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                )
                bar.next()
            if i % 100 == 0:
                del output, loss
                torch.cuda.empty_cache()
        if args.local_rank == 0:
            bar.finish()
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.local_rank == 0:
        progress.display_summary()
        print("current beta:\t", "\t".join("{:.6f} ({:.0f})".format(b.item(), round(b.item(), 0)) for b in find_beta(model)))
        print("current the:\t", "\t".join("{:.6f}".format(b.item()) for b in find_the(model)))
    return top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth', checkpoint_folder='.'):
    checkpoint_path = os.path.join(checkpoint_folder, filename)
    torch.save(state, checkpoint_path)
    if is_best:
        best_model_path = os.path.join(checkpoint_folder, 'model_best.pth')
        shutil.copyfile(checkpoint_path, best_model_path)
        print("Best model saved in {}".format(best_model_path))


class focal_loss(nn.Module):
    def __init__(self, alpha=None, gamma=2, num_classes=3, size_average=True):
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中第一类为背景类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

        print('Focal Loss:')
        print('    Alpha = {}'.format(self.alpha))
        print('    Gamma = {}'.format(self.gamma))

    def forward(self, preds, labels):

        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))
        alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1)  # log_softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(optimizer, epoch):
    if epoch in [30, 60, 90]:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


if __name__ == '__main__':
    args = parser.parse_args()
    if args.debug_mode:
        SummaryWriter = BlankSummaryWriter
        
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ.get("WORLD_SIZE", 1))

    args.distributed = args.world_size > 1

    if args.distributed or (args.dist_url == "env://" and args.rank == -1):
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = rank
        args.rank = rank
        args.local_rank = local_rank
        args.global_rank = global_rank
    else:
        warnings.warn(
            "It is detected that neither (world_size>1) nor (dist_url=='env://' and args.rank== -1), Please use 'python -m torch.distributed.run' to launch distributed training for automatically distributing.")
        rank = args.rank
        local_rank = args.local_rank
        global_rank = rank
        args.global_rank = global_rank

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, world_size=args.world_size, rank=args.rank)


    def print(*_args, **_kwargs):
        if local_rank <= 0 or not args.distributed:
            builtins.print(*_args, **_kwargs)
        else:
            pass


    print("==> Options:", args)
    main(args)
