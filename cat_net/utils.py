import math
import numpy as np
from PIL import Image

import torch
from torch import nn
from torchvision import transforms

from . import custom_transforms

from . import config


def print_network(net):
    num_params = sum([param.numel() for param in net.parameters()])
    print(net)
    print('Total number of parameters: {}'.format(num_params))


def initialize_weights(module):
    classname = module.__class__.__name__

    if classname == 'Conv2d' or classname == 'ConvTranspose2d':
        module.weight.data.normal_(0.0, 0.02)  # Isola et al. 2014
        if module.bias is not None:
            nn.init.constant(module.bias, 0.)

    elif 'Norm' in classname:
        if module.weight is not None:
            module.weight.data.normal_(1.0, 0.02)  # Isola et al. 2014
        if module.bias is not None:
            nn.init.constant(module.bias, 0.)


def image_from_tensor(tensor):
    """Scales a CxHxW tensor with values in the range [-1, 1] to [0, 255]"""
    image = tensor.cpu()
    image = 0.5 * image + 0.5  # [-1, 1] --> [0, 1]
    image = transforms.ToPILImage()(image)  # [0, 1] --> [0, 255]

    return image


def concatenate_dicts(*dicts):
    concat_dict = {}
    for key in dicts[0]:
        concat_dict[key] = []
        for d in dicts:
            val = d[key]
            if isinstance(val, list):
                concat_dict[key] = concat_dict[key] + val
            else:
                concat_dict[key].append(val)

    return concat_dict


def compute_dict_avg(dict):
    avg_dict = {}
    for key, val in dict.items():
        avg_dict[key] = np.mean(np.array(val))
    return avg_dict


def tag_dict_keys(dict, tag):
    new_dict = {}
    for key, val in dict.items():
        new_key = tag + '_' + key
        new_dict[new_key] = val
    return new_dict
