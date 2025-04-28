from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from micronet.compression.quantization.wqaq.dorefa import quantize
import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.nn import init
import torch.nn.functional as F
from utils import get_network, get_test_dataloader, get_training_dataloader, parse_device
from conf import settings


def setup_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_state(model, best_acc):
    print("==> Saving model ...")
    state = {
        "best_acc": best_acc,
        "state_dict": model.state_dict(),
    }
    state_copy = state["state_dict"].copy()
    for key in state_copy.keys():
        if "module" in key:
            state["state_dict"][key.replace("module.", "")] = state["state_dict"].pop(
                key
            )

    if args.prune_qat:
        torch.save(
            {"cfg": cfg, "best_acc": best_acc, "state_dict": state["state_dict"]},
            "models_save/dorefa_vanilla.pth",
        )
    else:
        torch.save(state, "./dorefa_vanilla.pth")


def adjust_learning_rate(optimizer, epoch):
    update_list = [30, 40]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * 0.5
    return


def train(epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(trainloader):
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}".format(
                    epoch,
                    batch_idx * len(data),
                    len(trainloader.dataset),
                    100.0 * batch_idx / len(trainloader),
                    loss.data.item(),
                    optimizer.param_groups[0]["lr"],
                )
            )
    return


def test():
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in testloader:
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    acc = 100.0 * float(correct) / len(testloader.dataset)

    if acc > best_acc:
        best_acc = acc
        save_state(model, best_acc)
    average_test_loss = test_loss / (len(testloader.dataset) / args.eval_batch_size)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
            average_test_loss,
            correct,
            len(testloader.dataset),
            100.0 * float(correct) / len(testloader.dataset),
        )
    )

    print("Best Accuracy: {:.2f}%\n".format(best_acc))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--net", action="store_true", default="resnet20", help="network name to train (or quant), ref to utils.py/get_network"
    )
    parser.add_argument("--device_id", action="store", default="",
                        help="device_id, eg: 0 for single #0 gpu or 0,1,2 for multi-gpus (#0, #1 and #2). Set any negtive num (eg: -1) if only CPU is available")
    parser.add_argument(
        "--lr", action="store", default=0.01, help="the intial learning rate"
    )
    parser.add_argument(
        "--wd", action="store", default=1e-5, help="weight decay"
    )
    # refine
    parser.add_argument(
        "--refine",
        default="",
        type=str,
        metavar="PATH",
        help="the path to the float_refine model",
    )
    # resume
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="the path to the resume model",
    )
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument(
        "--start_epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train_start",
    )
    parser.add_argument(
        "--end_epochs",
        type=int,
        default=60,
        metavar="N",
        help="number of epochs to train_end",
    )
    # W/A — bits
    parser.add_argument("--w_bits", type=int, default=8)
    parser.add_argument("--a_bits", type=int, default=8)
    args = parser.parse_args()
    print("==> Options:", args)

    device = parse_device(args.device_id)
    if device != "cpu":
        use_gpu = True
    else:
        use_gpu = False

    setup_seed(1)

    # data preprocessing:
    trainloader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=128,
        shuffle=True
    )

    testloader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=128,
        shuffle=True
    )

    print("******Initializing model******")
    print("==> Preparing data..")
    model = get_network(args)
    print(f"original model:\n{model}")
    best_acc = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                init.zeros_(m.bias)
    print("***ori_model***\n", model)
    quantize.prepare(model, inplace=True, a_bits=args.a_bits, w_bits=args.w_bits)
    print("\n***quant_model***\n", model)

    if use_gpu:
        model.cuda()
        model = torch.nn.DataParallel(
            model, device_ids=device
        )

    base_lr = float(args.lr)
    param_dict = dict(model.named_parameters())
    params = []
    for key, value in param_dict.items():
        params += [{"params": [value], "lr": base_lr, "weight_decay": args.wd}]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params, lr=base_lr, weight_decay=args.wd)

    for epoch in range(args.start_epochs, args.end_epochs):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test()
