import torch
from torch import nn
from torch.autograd import Variable

import os

from . import config
from . import utils
from . import networks


class CATModel:
    def __init__(self):
        self.use_cuda = config.use_cuda

        # Initialize network
        self.net = networks.UNet(source_channels=config.source_channels,
                                 output_channels=config.output_channels,
                                 down_levels=config.down_levels,
                                 num_init_features=config.num_init_features,
                                 max_features=config.max_features,
                                 drop_rate=config.drop_rate,
                                 innermost_kernel_size=config.innermost_kernel_size,
                                 use_cuda=self.use_cuda)

        if self.use_cuda:
            self.net.cuda()

        self.net.apply(utils.initialize_weights)

        print('\n{:-^50}'.format(' Network initialized '))
        utils.print_network(self.net)
        print('-' * 50 + '\n')

        # Set loss function
        self.loss_function = nn.MSELoss()

        print('\n{:-^50}'.format(' Loss initialized '))
        print(self.loss_function)
        print('-' * 50 + '\n')

        # Set optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          lr=config.lr)

    def set_mode(self, mode):
        """Set the network to train/eval mode. Affects dropout and batchnorm."""
        if mode == 'train':
            self.net.train()
        elif mode == 'eval':
            self.net.eval()
        else:
            raise ValueError(
                "Got invalid mode '{}'. Valid options are 'train' and 'eval'.".format(mode))

    def set_data(self, source, target):
        """Set the source and target tensors"""
        if self.use_cuda:
            self.source = source.cuda(async=True)
            self.target = target.cuda(async=True)
        else:
            self.source = source
            self.target = target

    def optimize(self):
        """Do one step of training with the current source and target tensors"""
        self.output = self.net.forward(Variable(self.source))
        self.optimizer.zero_grad()
        self.loss = self.loss_function(self.output, Variable(self.target))
        self.loss.backward()
        self.optimizer.step()

    def test(self):
        """Evaluate the model and test loss without optimizing"""
        self.output = self.net.forward(Variable(self.source, volatile=True))
        self.loss = self.loss_function(self.output,
                                       Variable(self.target, volatile=True))

    def get_images(self):
        """Return a dictionary of the current source/output/target images"""
        return {'source': utils.image_from_tensor(self.source[0]),
                'output': utils.image_from_tensor(self.output.data[0]),
                'target': utils.image_from_tensor(self.target[0])}

    def get_errors(self):
        """Return a dictionary of the current errors"""
        return {'loss': self.loss.data[0]}

    def save_checkpoint(self, label):
        """Save the model to file"""
        model_dir = os.path.join(
            config.results_dir, config.experiment_name, 'checkpoints')
        os.makedirs(model_dir, exist_ok=True)
        model_file = os.path.join(model_dir, '{}_net.pth.tar'.format(label))

        model_dict = {'net_state_dict': self.net.state_dict(),
                      'use_cuda': self.use_cuda}

        print("Saving model to {}".format(model_file))
        torch.save(model_dict, model_file)

    def load_checkpoint(self, label):
        """Load a model from file"""
        model_dir = os.path.join(
            config.results_dir, config.experiment_name, 'checkpoints')
        model_file = os.path.join(model_dir, '{}_net.pth.tar'.format(label))

        print("Loading model from {}".format(model_file))
        model_dict = torch.load(model_file)

        self.use_cuda = model_dict['use_cuda']
        if self.use_cuda:
            self.net.cuda()

        self.net.load_state_dict(model_dict['net_state_dict'])
