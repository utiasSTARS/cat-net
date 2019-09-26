import argparse
import os

from torch.utils.data import ConcatDataset

from cat_net.options import Options
from cat_net.models import CATModel
from cat_net.datasets import vkitti
from cat_net import experiment

### COMMAND LINE ARGUMENTS ###
parser = argparse.ArgumentParser()
parser.add_argument('stage', type=str, choices=['train', 'test', 'both'])
parser.add_argument('--resume', action='store_true')
args = parser.parse_args()

resume_from_epoch = 'latest' if args.resume else None

### CONFIGURATION ###
opts = Options()

opts.data_dir = '/media/m2-drive/datasets/virtual-kitti/raw'
opts.results_dir = '/media/raid5-array/experiments/cat-net/virtual-kitti'

opts.down_levels = 7
opts.innermost_kernel_size = (3, 4)

### SET TRAINING, VALIDATION AND TEST SETS ###
seqs = ['0001', '0002', '0006', '0018', '0020']
conds = ['clone', 'morning', 'overcast', 'sunset']
canonical = conds[2]

for test_seq in seqs:
    train_seqs = seqs.copy()
    train_seqs.remove(test_seq)
    val_seqs = [test_seq]
    val_conds = [conds[0]]

    train_data = []
    for seq in train_seqs:
        for cond in conds:
            print('Train {}: {} --> {}'.format(seq, cond, canonical))
            data = vkitti.TorchDataset(
                opts, seq, cond, canonical, opts.random_crop)
            train_data.append(data)
    train_data = ConcatDataset(train_data)

    val_data = []
    for seq in val_seqs:
        for cond in val_conds:
            print('Val {}: {} --> {}'.format(seq, cond, canonical))
            data = vkitti.TorchDataset(
                opts, seq, cond, canonical, False)
            val_data.append(data)
    val_data = ConcatDataset(val_data)

    ### TRAIN / TEST ###
    opts.experiment_name = '{}-test'.format(test_seq)
    model = CATModel(opts)

    if args.stage == 'train' or args.stage == 'both':
        print(opts)
        opts.save_txt()
        experiment.train(opts, model, train_data, val_data,
                         opts.train_epochs, resume_from_epoch=resume_from_epoch)

    if args.stage == 'test' or args.stage == 'both':
        for cond in conds:
            expdir = os.path.join(opts.experiment_name, '{}-test'.format(cond))
            test_data = vkitti.TorchDataset(opts, seq, cond, canonical, False)
            experiment.test(opts, model, test_data, expdir=expdir,
                            save_loss=True, save_images=True)
