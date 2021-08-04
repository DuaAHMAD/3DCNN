from __future__ import print_function
import numpy as np
import torch
import config


def BCELossWeighted(pred, target, weight):
    out = (-(target * torch.log(pred + 1e-40) + (1 - target) * torch.log(1 - pred + 1e-40))) * weight
    return torch.mean(out)


def image_to_gpu(img):
    """Put the Image tensor to gpu and reshape the Image to B x #face_images x C x H x W"""
    return exchange_temp_channel_axes(img.cuda().view(-1, config.LEN_SAMPLE, config.C, config.F_H, config.F_W)).float().div(255)

def label_to_gpu(label):
    return label[0].cuda()

def exchange_temp_channel_axes(tensor):
    return tensor.transpose(1, 2)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val*weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        val = np.asarray(val)
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val.tolist()

    def average(self):
        return self.avg.tolist()