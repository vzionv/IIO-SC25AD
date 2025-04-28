import random
import warnings
import torch
import copy
import numpy as np
from numpy import ndarray
import torch.nn as nn
from torch import Tensor

MINSCALE = 0.25
MAXSCALE = 1.25


def apply_random_flip(input, flip_rate=0.5, dim=0):
    """
    Args:
        input <torch.Tensot>: (*)
    Returns:
        output <torch.Tensot>: (*)
    """
    if type(dim) is int:
        dim = (dim,)

    flip = random.random() < flip_rate

    if flip:
        output = torch.flip(input, dims=dim)
    else:
        output = input

    return output


class RandomFlip:
    def __init__(self, flip_rate=0.5, dim=0):
        self.flip_rate = flip_rate
        self.dim = dim

    def __call__(self, input):
        output = apply_random_flip(input, flip_rate=self.flip_rate, dim=self.dim)
        return output


def apply_random_gain(input, min=MINSCALE, max=MAXSCALE):
    """
    Args:
        input <torch.Tensot>: (*)
    Returns:
        output <torch.Tensot>: (*)
    """
    scale = random.uniform(min, max)
    output = scale * input

    return output


class RandomGain:
    def __init__(self, min=MINSCALE, max=MAXSCALE):
        self.min, self.max = min, max

    def __call__(self, input):
        output = apply_random_gain(input, min=self.min, max=self.max)
        return output


def apply_random_sign(input, rate=0.5):
    """
    Args:
        input <torch.Tensot>: (*)
    Returns:
        output <torch.Tensot>: (*)
    """
    if random.random() < rate:
        sign = -1
    else:
        sign = 1

    output = sign * input

    return output


class RandomSign:
    def __init__(self, rate=0.5):
        self.rate = rate

    def __call__(self, input):
        output = apply_random_sign(input, rate=self.rate)
        return output


# ========================================================


# class ToTensor(object):
#     def __call__(self, sample):
#         text = sample['IEGM_seg']
#         return {
#             'IEGM_seg': torch.from_numpy(text),
#             'label': sample['label']
#         }


# class ToTensor(object):
#     def __call__(self, sample):
#         return torch.from_numpy(sample)


# class DownsampleToTensor(object):
#     def __call__(self, sample):
#         sample = np.mean(sample.reshape((-1, 2)), axis=1).reshape((1, -1, 1))
#         return torch.from_numpy(sample)


class RandomMix:
    def __init__(self, sampling_rate, duration, seed=None):
        super(RandomMix, self).__init__()
        self.sampling_rate = sampling_rate
        self.min_duration = self.max_duration = self.duration = None
        try:
            self.min_duration, self.max_duration = duration
            assert self.min_duration < self.max_duration, "min_duration should be smaller than max_duration"
        except Exception:
            self.duration = duration
        self.seed = seed

    def __call__(self, *sources):
        if self.duration is None:
            cut_n = torch.randint(int(self.min_duration * self.sampling_rate), int(self.max_duration * self.sampling_rate), (1,))
        else:
            cut_n = int(self.duration * self.sampling_rate)

        sources_t = None
        mixture = 0
        for source in sources:
            source: torch.Tensor
            n_frames = source.shape[-1]
            if n_frames <= cut_n:
                source = source.repeat(*source.shape[:-1], cut_n // n_frames + 1)
            random_start = torch.randint(0, source.shape[-1] - cut_n, (1,))
            random_ratio = random.uniform(0.5, 1.0)
            source = source[..., random_start:random_start + cut_n] * random_ratio
            mixture += source
            if sources_t is None:
                sources_t = source
            else:
                sources_t = torch.cat([sources_t, source], dim=0)
        return mixture, sources_t


def minmax_normalize(inputs: ndarray, new_min=-1, new_max=1):
    old_min, old_max = inputs.min(), inputs.max()
    if old_max - old_min:
        return (inputs - old_min) * (new_max - new_min) / (old_max - old_min) + new_min
    else:
        return inputs


def minmax_normalize_DC(inputs: ndarray):
    mu = np.mean(inputs, axis=-1, keepdims=True)
    old_min, old_max = inputs.min(), inputs.max()
    outputs = (inputs - mu) / (old_max - old_min + 1E-6)
    return outputs


class MinMaxNorm:
    def __init__(self, new_min=-1, new_max=1):
        super().__init__()
        self.new_min = new_min
        self.new_max = new_max

    def __call__(self, inputs: Tensor):
        # old_min, old_max = inputs.min(), inputs.max()
        old_min, old_max = -2.1179, 2.6400
        if old_max - old_min:
            return (inputs - old_min) / (old_max - old_min) * (self.new_max - self.new_min) + self.new_min
        else:
            return inputs


def zscore_normalize(inputs: ndarray):
    mu = np.mean(inputs, axis=-1, keepdims=True)
    sigma = np.std(inputs, axis=-1, keepdims=True)
    outputs = (inputs - mu) / (sigma + 1E-6)
    return outputs


class ZScoreNorm:
    def __call__(self, inputs: Tensor):
        mu = inputs.mean(dim=-1, keepdim=True)
        sigma = inputs.std(dim=-1, keepdim=True)
        outputs = (inputs - mu) / (sigma + 1E-6)
        return outputs


if __name__ == '__main__':
    pass
