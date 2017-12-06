import numpy as np
from visdom import Visdom

from . import config


class Visualizer:
    def __init__(self):
        self.vis = Visdom(port=config.visdom_port)
        self.plot_data = {'x': [], 'y': [], 'legend': []}

    def show_images(self, images):
        for label, image in images.items():
            self.vis.image(np.array(image).transpose([2, 0, 1]),
                           opts={'title': label},
                           win=label)

    def plot_errors(self, epoch, errors):
        if len(self.plot_data['legend']) == 0:
            self.plot_data['legend'] = sorted(list(errors.keys()))
        self.plot_data['x'].append([epoch for _ in self.plot_data['legend']])
        self.plot_data['y'].append([errors[key]
                                    for key in self.plot_data['legend']])

        self.vis.line(
            X=np.array(self.plot_data['x']),
            Y=np.array(self.plot_data['y']),
            opts={'title': '{} loss'.format(config.experiment_name),
                  'legend': self.plot_data['legend'],
                  'xlabel': 'epoch',
                  'ylabel': 'loss'},
            win='loss'
        )

    def print_errors(self, errors):
        for key, val in errors.items():
            print('{:20}: {:.3e}'.format(key, val))
