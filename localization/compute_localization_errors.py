import os
import re

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pyslam.metrics import TrajectoryMetrics
from pyslam.visualizers import TrajectoryVisualizer


def print_file(data_dir, file):
    tm = TrajectoryMetrics.loadmat(os.path.join(data_dir, file))

    tokens = re.split('\.|-', file)
    trans_err, rot_err = tm.mean_err(rot_unit='deg')
    length = tm.cum_dists[-1]

    if 'vo' in tokens:
        ref_seq = tokens[0] + '-' + tokens[1]
        track_seq = ref_seq
    else:
        nonflag_tokens = [t for t in tokens if t != 'cat']
        ref_seq = nonflag_tokens[0] + '-' + tokens[1]
        track_seq = nonflag_tokens[0] + '-' + tokens[2]

    if 'cat' in tokens:
        ref_seq += ' (CAT)'
        track_seq += ' (CAT)'

    print('{:20} -> {:20} | {:7d} {:7.2f} m | {:7.2f} m {:7.2f} deg'.format(
        ref_seq, track_seq, tm.num_poses, length, trans_err, rot_err))


def make_plots(data_dir, file, label, Twv_gt=None):
    file_tokens = file.split('.')
    file_cat = file_tokens[0] + '-cat.' + file_tokens[1]

    tm_dict = {label: TrajectoryMetrics.loadmat(
        os.path.join(data_dir, file))}

    if os.path.exists(os.path.join(data_dir, file_cat)):
        tm_dict.update(
            {label + ' + CAT': TrajectoryMetrics.loadmat(os.path.join(data_dir, file_cat))})

    vis = TrajectoryVisualizer(tm_dict)

    try:
        f, ax = vis.plot_norm_err(outfile=os.path.join(
            data_dir, file_tokens[0] + '_norm_err.pdf'), figsize=(6, 2), tight_layout=True, fontsize=12, err_xlabel='Frame')
        plt.close(f)
    except Exception as e:
        print(e)

    try:
        f, ax = vis.plot_segment_errors(segs, outfile=os.path.join(
            data_dir, file_tokens[0] + '_seg_err.pdf'), figsize=(6, 2), tight_layout=True, fontsize=12, err_xlabel='Frame')
        plt.close(f)
    except Exception as e:
        print(e)

    try:
        if Twv_gt is not None:
            tm_dict[label].Twv_gt = Twv_gt

        f, ax = vis.plot_topdown(topdown_plane, outfile=os.path.join(
            data_dir, file_tokens[0] + '_topdown.pdf'), figsize=(4, 3), tight_layout=True, use_endpoint_markers=True, fontsize=12)
        plt.close(f)
    except Exception as e:
        print(e)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    required=True, help='which dataset?')
parser.add_argument('--print', action='store_true', help='print errors?')
parser.add_argument('--plot', action='store_true', help='plot errors?')
parser.add_argument('--no_vo', action='store_true', help='skip VO?')
parser.add_argument('--no_reloc', action='store_true',
                    help='skip relocalization?')
args = parser.parse_args()

data_dir = '/media/raid5-array/experiments/appearance-transfer/virtual-kitti/vkitti-to-kitti/pyslam'

if args.dataset == 'ethl_dataset':
    topdown_plane = 'xy'
    segs = [0.25, 0.5, 1., 2., 3.]
elif args.dataset == 'virtual-kitti':
    topdown_plane = 'xz'
    segs = [25., 50., 100., 200., 300.]
elif args.dataset == 'kitti_affine':
    topdown_plane = 'xy'
    segs = [100., 200., 400., 600., 800.]

data_files = [f for f in os.listdir(data_dir) if
              f.split('.')[-1] == 'mat' and f[0] != '.']
vo_files = sorted([f for f in data_files if '-vo' in f])
reloc_files = sorted([f for f in data_files if '-vo' not in f])

if args.print:
    print('{:20} -> {:20} | {:>7} {:>9} | {:>9} {:>11}'.format('Reference', 'Tracking',
                                                               'Poses', 'Length',
                                                               'Tran.', 'Rot.'))
    print('-' * 89)

    # Print VO results
    if not args.no_vo:
        for file in vo_files:
            print_file(data_dir, file)

        print('-' * 89)

    # Print relocalization results
    if not args.no_reloc:
        for file in reloc_files:
            print_file(data_dir, file)

if args.plot:
    # Make comparison plots:
    if not args.no_vo:
        for file in vo_files:
            make_plots(data_dir, file, 'VO')

    if not args.no_reloc:
        for file in reloc_files:
            gt_seq_name = file.split('-')[0]
            tm_gt = TrajectoryMetrics.loadmat(
                os.path.join(data_dir, gt_seq_name + '-vo.mat'))
            make_plots(data_dir, file, 'Reloc.', Twv_gt=tm_gt.Twv_gt)
