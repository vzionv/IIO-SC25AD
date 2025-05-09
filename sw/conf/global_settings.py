""" configurations for this project

author baiyu
"""
import os
from datetime import datetime

# CIFAR100 dataset path (python version)

# mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

CIFAR10_TRAIN_MEAN = (0.49139968, 0.48215841, 0.44653091)
CIFAR10_TRAIN_STD = (0.24703223, 0.24348513, 0.26158784)

# CIFAR100_TEST_MEAN = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
# CIFAR100_TEST_STD = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)

# directory to save weights file
CHECKPOINT_PATH = 'checkpoint'

# total training epoches
EPOCH = 60
MILESTONES = [20, 30, 40, 50]

# initial learning rate
# INIT_LR = 0.1

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
# time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

# tensorboard log dir
LOG_DIR = 'runs'

# save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10
