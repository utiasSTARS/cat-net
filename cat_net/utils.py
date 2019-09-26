import math
import numpy as np
from PIL import Image

import torch
from torch import nn
from torchvision import transforms

from . import transforms as custom_transforms


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


def image_from_tensor(tensor, image_mean=0., image_std=1., size=None):
    tensor = tensor.cpu()

    if tensor.size(0) == 1:
        # need to make a copy here for unnormalization to work right
        tensor = tensor.repeat(3, 1, 1)

    tf_list = [custom_transforms.UnNormalize(image_mean, image_std),
               custom_transforms.Clamp(0, 1),
               transforms.ToPILImage()]  # multiplication by 255 happens here

    if size is not None:
        tf_list.append(transforms.Resize(size, interpolation=Image.NEAREST))

    transform = transforms.Compose(tf_list)

    return transform(tensor)


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
        new_key = key + '/' + tag
        new_dict[new_key] = val
    return new_dict
