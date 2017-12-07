from torch.utils.data import DataLoader, ConcatDataset

from cat_net import config
from cat_net.models import CATModel
from cat_net.datasets import tum_rgbd, vkitti
from cat_net import experiment

### CONFIGURATION ###
### (defaults in __init__.py) ###
config.data_dir = '/home/leeclement/datasets/ethl_dataset'
config.results_dir = '/home/leeclement/experiments/cat-net/ethl_dataset'

config.experiment_name = 'ethl-demo'

config.use_cuda = True
config.down_levels = 7
config.innermost_kernel_size = (3, 4)


print(config)
config.save_txt()


### INITIALIZE MODEL ###
model = CATModel()


### TRAIN AND VALIDATE ###
train_canonical = 'ethl1'
train_data = ConcatDataset(
    [tum_rgbd.TorchDataset('ethl1', train_canonical),
     tum_rgbd.TorchDataset('ethl1_global', train_canonical),
     tum_rgbd.TorchDataset('ethl1_local', train_canonical),
     tum_rgbd.TorchDataset('ethl1_loc_glo', train_canonical),
     tum_rgbd.TorchDataset('ethl1_flash', train_canonical)])

val_canonical = 'ethl2'
val_data = tum_rgbd.TorchDataset('ethl2_local', val_canonical)

experiment.train(model, train_data, val_data)


### TEST ###
test_seqs = ['ethl2', 'ethl2_global',
             'ethl2_local', 'ethl2_loc_glo', 'ethl2_flash']
test_canonical = 'ethl2'

for seq in test_seqs:
    test_data = tum_rgbd.TorchDataset(seq, test_canonical)
    experiment.test(model, test_data, label=seq)
