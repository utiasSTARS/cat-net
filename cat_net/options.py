import torch
import numpy as np
import os


class Options:
    """Container class to store configuration parameters, 
        plus common defaults.
    """

    def __init__(self):
        self.data_dir = 'data'
        self.results_dir = 'results'
        self.experiment_name = 'experiment'

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataloader_workers = 6

        self.train_epochs = 100
        self.lr = 1e-4

        self.batch_size = 64
        self.source_channels = 3
        self.output_channels = 3
        self.num_init_features = 64
        self.max_features = 512
        self.drop_rate = 0.5

        # Number of times to downsample by 2 in the network
        self.down_levels = 7
        # Should be the aspect ratio of the input image for 1x1 bottleneck layer
        self.innermost_kernel_size = (3, 4)

        self.image_load_size = (240, 320)  # H, W
        self.image_final_size = (192, 256)  # H, W
        self.random_crop = True  # if True, crops load_size to final_size, else scales

        # mean and std for normalizing images
        # used by Isola et al. (2014) to play nice with final tanh layer
        self.image_mean = [0.5, 0.5, 0.5]
        self.image_std = [0.5, 0.5, 0.5]

        self.checkpoint_interval = 5  # epochs

        self.set_random_seed(404)

    def set_random_seed(self, seed):
        self.random_seed = seed
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

    def save_txt(self):
        save_dir = os.path.join(self.results_dir, self.experiment_name)
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, 'config.txt')

        print("Saving config to {}".format(save_file))
        with open(save_file, 'wt') as file:
            file.write(self.__repr__())

    def to_dict(self):
        return vars(self)

    def from_dict(self, dict):
        for key, val in dict.items():
            setattr(self, key, val)

    def __repr__(self):
        args = vars(self)
        string = '\n{:-^50}\n'.format(' Options ')
        for key, val in sorted(args.items()):
            string += ' {:25}: {}\n'.format(str(key), str(val))
        string += '-' * 50 + '\n'
        return string
