from torch.utils.data import DataLoader, ConcatDataset

from cat_net import config
from cat_net.models import CATModel
from cat_net.datasets import tum_rgbd, vkitti
from cat_net import experiment

### CONFIGURATION ###
### (defaults in __init__.py) ###
# config.data_dir = '/media/m2-drive/datasets/ethl_dataset/raw'
# config.results_dir = '/media/raid5-array/experiments/cat-net/ethl_dataset'
config.data_dir = '/media/m2-drive/datasets/virtual-kitti/raw'
config.results_dir = '/media/raid5-array/experiments/cat-net/virtual-kitti'

config.experiment_name = 'vkitti-all'

config.use_cuda = True
config.down_levels = 7
config.innermost_kernel_size = (3, 4)
config.visdom_port = 8097


print(config)
config.save_txt()


### INITIALIZE MODEL ###
model = CATModel()


### TRAIN AND VALIDATE ###
# train_canonical = 'ethl2'
# train_data = ConcatDataset(
#     [tum_rgbd.TorchDataset('ethl2', 'train_canonical'),
#      tum_rgbd.TorchDataset('ethl2_global', train_canonical),
#      tum_rgbd.TorchDataset('ethl2_local', train_canonical),
#      tum_rgbd.TorchDataset('ethl2_loc_glo', train_canonical),
#      tum_rgbd.TorchDataset('ethl2_flash', train_canonical)])

# train_data = ConcatDataset(
#     [tum_rgbd.TorchDataset('ethl1', 'ethl1'),
#      tum_rgbd.TorchDataset('ethl1_global', 'ethl1'),
#      tum_rgbd.TorchDataset('ethl1_local', 'ethl1'),
#      tum_rgbd.TorchDataset('ethl1_loc_glo', 'ethl1'),
#      tum_rgbd.TorchDataset('ethl1_flash', 'ethl1'),
#      tum_rgbd.TorchDataset('ethl2', 'ethl2'),
#      tum_rgbd.TorchDataset('ethl2_global', 'ethl2'),
#      tum_rgbd.TorchDataset('ethl2_local', 'ethl2'),
#      tum_rgbd.TorchDataset('ethl2_loc_glo', 'ethl2'),
#      tum_rgbd.TorchDataset('ethl2_flash', 'ethl2')])

train_seqs = ['0001', '0002', '0006', '0018', '0020']
conds = ['clone', 'morning', 'overcast', 'sunset']
canonical = 'overcast'

train_data = []
for seq in train_seqs:
    for cond in conds:
        train_data.append(vkitti.TorchDataset(seq, cond, canonical))
train_data = ConcatDataset(train_data)

# val_canonical = 'ethl1'
# val_data = tum_rgbd.TorchDataset('ethl1_local', val_canonical)

val_data = vkitti.TorchDataset('0018', 'clone', canonical)

experiment.train(model, train_data, val_data)


### TEST ###
# test_seqs = ['ethl1', 'ethl1_global',
#              'ethl1_local', 'ethl1_loc_glo', 'ethl1_flash']
# test_canonical = 'ethl1'

# for seq in test_seqs:
#     test_data = tum_rgbd.TorchDataset(seq, test_canonical)
#     experiment.test(model, test_data, label=seq)

# test_seqs = ['real_global', 'real_local', 'real_flash']
# for seq in test_seqs:
#     test_data = tum_rgbd.TorchDataset(seq, seq)
#     experiment.test(model, test_data, label=seq)

# test_seqs = ['0018']
# test_data = []
# for seq in test_seqs:
#     for cond in conds:
#         test_data = vkitti.TorchDataset(seq, cond, canonical)
#         experiment.test(model, test_data, label=(seq + '-' + cond))
