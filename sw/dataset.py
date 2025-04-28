# -*- coding: utf-8 -*-
import os
import sys
import pickle

from skimage import io
import matplotlib.pyplot as plt
import numpy
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import math


class DistributedRandomSampler(DistributedSampler):
    def __init__(
            self,
            dataset,
            num_samples=None,
            num_replicas=None,
            rank=None,
            seed=0,
    ):
        super().__init__(
            dataset,
            num_replicas,
            rank,
            seed,
        )
        self.total_size = num_samples
        self.num_samples_per_rank = math.ceil(self.total_size / self.num_replicas)
        assert self.total_size <= len(self.dataset)

    def __iter__(self):
        # deterministically shuffle based on epoch and seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
        indices = indices[: self.total_size + self.rank]

        # builtins.print(f"in rank {rank}", self.num_replicas)
        # subsample
        indices = indices[self.rank: self.total_size + self.rank: self.num_replicas]
        # builtins.print(f"in rank {rank}", indices[0], indices[-1])
        # builtins.print(f"in rank {rank}", len(indices), self.num_samples_per_rank)
        assert len(indices) == self.num_samples_per_rank

        return iter(indices)

    def __len__(self):
        return self.num_samples_per_rank


# class DistributedRandomSampler(torch.utils.data.Sampler):
#     def __init__(self, dataset, num_samples,
#                  num_replicas = None, rank = None,
#                  seed = 0):
#         """ Sample elements randomly with distributed learning.
#
#         Args:
#             dataset: Dataset to sample from.
#             num_samples: Number of samples to draw.
#             num_replicas: Number of processes participating in distributed training.
#                 By default, :attr:`world_size` is retrieved from the current distributed group.
#             rank: Rank of the current process within :attr:`num_replicas`.
#                 By default, :attr:`rank` is retrieved from the current distributed group.
#             seed: Random seed used to shuffle the sampler.
#                 This number should be identical across all processes in the distributed group.
#         """
#         if num_replicas is None:
#             if not torch.distributed.is_available():
#                 raise RuntimeError("Requires distributed package to be available")
#             num_replicas = torch.distributed.get_world_size()
#         if rank is None:
#             if not torch.distributed.is_available():
#                 raise RuntimeError("Requires distributed package to be available")
#             rank = torch.distributed.get_rank()
#         assert isinstance(rank, int) and isinstance(num_replicas, int)
#         assert isinstance(num_samples, int) and num_samples > 0, \
#             f"`num_samples` should be a positive integer value, got {num_samples}"
#
#         if rank >= num_replicas or rank < 0:
#             raise ValueError(f"Invalid rank={rank}, should be in the interval [0, {num_replicas - 1}]")
#
#         self.dataset = dataset
#         self.num_replicas = num_replicas
#         self.rank = rank
#         self.epoch = 0
#         self.seed = seed
#
#         self.num_samples = math.ceil(num_samples / self.num_replicas)
#         self._psize = 64
#
#     def __iter__(self) -> Iterator[int]:
#         # deterministically shuffle based on epoch
#         g = torch.Generator()
#         g.manual_seed(self.seed + self.epoch)
#
#         dataset_len = len(self.dataset)  # type: ignore
#         indices = torch.randint(0, dataset_len, (self.num_samples * self.num_replicas,),
#                                 dtype=torch.int64, generator=g)
#         yield from indices.tolist()[self.rank::self.num_replicas]
#
#     def __len__(self) -> int:
#         return self.num_samples
#
#     def set_epoch(self, epoch: int):
#         """ Sets the epoch for this sampler. This ensures all replicas use a different
#         random ordering for each epoch. Otherwise, the next iteration of this sampler
#         will yield the same ordering. """
#         self.epoch = epoch


class CIFAR100Train(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        # if transform is given, we transoform data using
        with open(os.path.join(path, 'train'), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform

    def __len__(self):
        return len(self.data['fine_labels'.encode()])

    def __getitem__(self, index):
        label = self.data['fine_labels'.encode()][index]
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = numpy.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        return label, image


class CIFAR100Test(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        with open(os.path.join(path, 'test'), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform

    def __len__(self):
        return len(self.data['data'.encode()])

    def __getitem__(self, index):
        label = self.data['fine_labels'.encode()][index]
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = numpy.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        return label, image
