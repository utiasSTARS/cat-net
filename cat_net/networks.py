import torch
from torch import nn


class UNet(nn.Module):
    """Create a U-Net with skip connections."""

    def __init__(self, source_channels, output_channels, down_levels,
                 num_init_features=64, max_features=512, drop_rate=0,
                 innermost_kernel_size=None,
                 use_cuda=False):
        super().__init__()

        self.use_cuda = use_cuda

        # Initial convolution
        self.model = nn.Sequential()
        self.model.add_module('conv0',
                              nn.Conv2d(source_channels, num_init_features,
                                        kernel_size=4, stride=2, padding=1,
                                        bias=True))
        down_levels = down_levels - 1  # We just downsampled one level
        total_features = num_init_features

        # Build the inner blocks recursively
        if down_levels > 0:
            submodule = SkipConnectionBlock(
                num_input_features=num_init_features,
                down_levels=down_levels,
                max_features=max_features,
                drop_rate=drop_rate,
                innermost_kernel_size=innermost_kernel_size)
            self.model.add_module('submodule', submodule)
            total_features += submodule.num_outer_features

        # Final convolution
        self.model.add_module('norm0',
                              nn.InstanceNorm2d(total_features, affine=True))
        self.model.add_module('relu0', nn.LeakyReLU(0.2, inplace=True))
        self.model.add_module('conv1',
                              nn.ConvTranspose2d(total_features,
                                                 output_channels,
                                                 kernel_size=4, stride=2, padding=1, output_padding=0,
                                                 bias=True))
        self.model.add_module('tanh', nn.Tanh())

        if self.use_cuda:
            self.model = nn.DataParallel(self.model)

    def forward(self, x):
        return self.model(x)


class SkipConnectionBlock(nn.Sequential):
    def __init__(self, num_input_features, down_levels,
                 max_features, drop_rate, innermost_kernel_size):
        super().__init__()

        self.num_outer_features = num_input_features
        self.num_inner_features = min(num_input_features * 2, max_features)

        if down_levels == 1 and innermost_kernel_size is not None:
            # This is the innermost block
            kernel_size = innermost_kernel_size
        else:
            kernel_size = 4

        # Downsampling
        self.add_module('norm0',
                        nn.InstanceNorm2d(self.num_outer_features, affine=True))
        self.add_module('relu0', nn.LeakyReLU(0.2, inplace=True))
        self.add_module('conv0',
                        nn.Conv2d(self.num_outer_features,
                                  self.num_inner_features,
                                  kernel_size=kernel_size, stride=2, padding=1,
                                  bias=True))
        if drop_rate > 0 and self.num_inner_features == self.num_outer_features:
            self.add_module('dropout0', nn.Dropout2d(drop_rate))

        down_levels = down_levels - 1  # We just downsampled one level

        # Submodule
        if down_levels > 0:
            submodule = SkipConnectionBlock(
                num_input_features=self.num_inner_features,
                down_levels=down_levels,
                max_features=max_features,
                drop_rate=drop_rate,
                innermost_kernel_size=innermost_kernel_size)

            self.add_module('submodule', submodule)
            total_features = self.num_inner_features + \
                submodule.num_outer_features
        else:
            total_features = self.num_inner_features

        # Upsampling
        self.add_module('norm1',
                        nn.InstanceNorm2d(total_features, affine=True))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1',
                        nn.ConvTranspose2d(total_features,
                                           self.num_outer_features,
                                           kernel_size=kernel_size, stride=2, padding=1, output_padding=0,
                                           bias=True))
        if drop_rate > 0 and self.num_inner_features == self.num_outer_features:
            self.add_module('dropout1', nn.Dropout2d(drop_rate))

    def forward(self, x):
        new_features = super().forward(x)
        return torch.cat([x, new_features], dim=1)
