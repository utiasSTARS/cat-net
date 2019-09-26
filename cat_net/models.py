import torch
from torch import nn
from torch.autograd import Variable

import os

from . import utils
from . import networks


class CATModel:
    def __init__(self, opts):
        self.opts = opts

        self.device = torch.device(self.opts.device)

        # Initialize network
        self.net = (networks.UNet(source_channels=self.opts.source_channels,
                                  output_channels=self.opts.output_channels,
                                  down_levels=self.opts.down_levels,
                                  num_init_features=self.opts.num_init_features,
                                  max_features=self.opts.max_features,
                                  drop_rate=self.opts.drop_rate,
                                  innermost_kernel_size=self.opts.innermost_kernel_size
                                  )).to(self.device)

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
                                          lr=self.opts.lr)

    def set_mode(self, mode):
        """Set the network to train/eval mode. Affects dropout and batchnorm."""
        if mode == 'train':
            self.net.train()
        elif mode == 'eval':
            self.net.eval()
        else:
            raise ValueError(
                "Got invalid mode '{}'. Valid options are 'train' and 'eval'.".format(mode))

    def set_data(self, data):
        """Set the source and target tensors"""
        self.source = data['source'].to(self.device)
        self.target = data['target'].to(self.device)

    def forward(self, compute_loss=True):
        """Evaluate the forward pass of the model"""
        self.output = self.net.forward(self.source)

        if compute_loss:
            self.loss = self.loss_function(self.output, self.target)

    def optimize(self):
        """Do one step of training with the current source and target tensors"""
        self.forward()
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def test(self, compute_loss=True):
        """Evaluate the model and test loss without optimizing"""
        with torch.no_grad():
            self.forward(compute_loss)

    def get_images(self):
        """Return a dictionary of the current source/output/target images"""
        return {'source': utils.image_from_tensor(self.source[0].detach(),
                                                  self.opts.image_mean,
                                                  self.opts.image_std),
                'output': utils.image_from_tensor(self.output[0].detach(),
                                                  self.opts.image_mean,
                                                  self.opts.image_std),
                'target': utils.image_from_tensor(self.target[0].detach(),
                                                  self.opts.image_mean,
                                                  self.opts.image_std)}

    def get_errors(self):
        """Return a dictionary of the current errors"""
        return {'loss': self.loss.item()}

    def save_checkpoint(self, epoch, label):
        """Save the model to file"""
        model_dir = os.path.join(
            self.opts.results_dir, self.opts.experiment_name, 'checkpoints')
        os.makedirs(model_dir, exist_ok=True)
        model_file = os.path.join(model_dir, '{}_net.pth.tar'.format(label))

        model_dict = {'epoch': epoch,
                      'label': label,
                      'net_state_dict': self.net.state_dict()}

        print("Saving model to {}".format(model_file))
        torch.save(model_dict, model_file)

    def load_checkpoint(self, label):
        """Load a model from file"""
        model_dir = os.path.join(
            self.opts.results_dir, self.opts.experiment_name, 'checkpoints')
        model_file = os.path.join(model_dir, '{}_net.pth.tar'.format(label))

        print("Loading model from {}".format(model_file))
        model_dict = torch.load(model_file, map_location=self.device)
        self.net.to(self.device)

        self.net.load_state_dict(model_dict['net_state_dict'])

        return model_dict['epoch']
