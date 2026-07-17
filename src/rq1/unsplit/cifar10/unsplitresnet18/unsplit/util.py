from __future__ import annotations

import random

import numpy as np
import torch


# Official UnSplit-style total variation.
def TV(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:, :, 1:, :])
    count_w = _tensor_size(x[:, :, :, 1:])
    h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, : h_x - 1, :], 2).sum()
    w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, : w_x - 1], 2).sum()
    return (h_tv / count_h + w_tv / count_w) / batch_size


def l2loss(x):
    return (x ** 2).mean()


def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]


def normalize(result):
    min_v = torch.min(result)
    range_v = torch.max(result) - min_v
    if range_v > 0:
        normalized = (result - min_v) / range_v
    else:
        normalized = torch.zeros(result.size(), device=result.device, dtype=result.dtype)
    return normalized


def get_random_indices(dataset, count=1, seed=123):
    rng = random.Random(seed)
    return rng.sample(range(len(dataset)), count)
