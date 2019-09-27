import ipdb
from pyslam.visualizers import TrajectoryVisualizer
from pyslam.metrics import TrajectoryMetrics
import matplotlib.pyplot as plt
import os
import re

import argparse
import matplotlib
matplotlib.use('Agg')


def entry_from_file(data_dir, file):
    tm = TrajectoryMetrics.loadmat(os.path.join(data_dir, file))

    tokens = re.split('; |\-|\.', file)
    trans_err, rot_err = tm.mean_err(rot_unit='deg')
    length = tm.cum_dists[-1]

    ref_seq = tokens[0]
    track_seq = ref_seq if 'vo' in tokens else tokens[1]

    if 'cat' in tokens:
        ref_seq += ' (CAT)'
        track_seq += ' (CAT)'

    line = '{},{},{},{},{},{}'.format(
        ref_seq, track_seq, tm.num_poses, length, trans_err, rot_err)
    return line


def make_plots(data_dir, file_prefix, segs, topdown_plane='xy', Twv_gt=None):
    file_rgb = file_prefix + '-rgb.mat'
    file_cat = file_prefix + '-cat.mat'

    tm_dict = {
        'Original': TrajectoryMetrics.loadmat(
            os.path.join(data_dir, file_rgb)),
        'CAT': TrajectoryMetrics.loadmat(
            os.path.join(data_dir, file_cat))
    }

    vis = TrajectoryVisualizer(tm_dict)

    try:
        outfile = os.path.join(
            data_dir, '{}_{}'.format(file_prefix, 'norm_err.pdf'))
        f, ax = vis.plot_norm_err(outfile=outfile,
                                  figsize=(9, 2.5), tight_layout=False, fontsize=12, err_xlabel='Frame')
        plt.close(f)
    except Exception as e:
        print(e)

    try:
        outfile = os.path.join(
            data_dir, '{}_{}'.format(file_prefix, 'seg_err.pdf'))
        f, ax = vis.plot_segment_errors(segs, outfile=outfile,
                                        figsize=(9, 2.5), tight_layout=False, fontsize=12, err_xlabel='Frame')
        plt.close(f)
    except Exception as e:
        print(e)

    try:
        if Twv_gt is not None:
            tm_dict['Original'].Twv_gt = Twv_gt

        outfile = os.path.join(
            data_dir, '{}_{}'.format(file_prefix, 'topdown.pdf'))
        f, ax = vis.plot_topdown(topdown_plane, outfile=outfile,
                                 figsize=(4, 3), tight_layout=False, use_endpoint_markers=True, fontsize=12)
        plt.close(f)
    except Exception as e:
        print(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='which dataset?', choices=[
        'ethl_dataset', 'virtual-kitti'])
    parser.add_argument('--no_vo', action='store_true', help='skip VO?')
    parser.add_argument('--no_reloc', action='store_true',
                        help='skip relocalization?')
    args = parser.parse_args()

    data_dir = '/media/raid5-array/experiments/cat-net/{}/pyslam'.format(
        args.dataset)

    if args.dataset == 'ethl_dataset':
        topdown_plane = 'xy'
        segs = [0.25, 0.5, 1., 2., 3.]  # meters
    elif args.dataset == 'virtual-kitti':
        topdown_plane = 'xz'
        segs = [25., 50., 100., 200., 300.]  # meters
    elif args.dataset == 'affine-kitti':
        topdown_plane = 'xy'
        segs = [100., 200., 400., 600., 800.]  # meters

    file_tags = ['rgb', 'cat']
    csv_header = ['Reference', 'Tracking', 'Poses', 'Length',
                  'Avg Trans Error (m)', 'Avg Rot Error (deg)']

    seq_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir)]

    for seq_dir in seq_dirs:
        rgb_files = [f for f in os.listdir(seq_dir) if
                     f.split('.')[-1] == 'mat' and f[0] != '.' and 'rgb' in f]
        file_prefixes = ['-'.join(f.split('-')[:-1]) for f in rgb_files]
        vo_file_prefixes = sorted([f for f in file_prefixes if '-vo' in f])
        reloc_file_prefixes = sorted(
            [f for f in file_prefixes if '-vo' not in f])

        if not args.no_vo:
            outfile = os.path.join(seq_dir, 'vo.csv')
            print(outfile)
            with open(outfile, 'w') as f:
                f.write(','.join(csv_header) + '\n')
                for vo_file_prefix in vo_file_prefixes:
                    for tag in file_tags:
                        vo_file = '{}-{}.mat'.format(vo_file_prefix, tag)
                        line = entry_from_file(seq_dir, vo_file)
                        f.write(line + '\n')

                    make_plots(seq_dir, vo_file_prefix, segs, topdown_plane)

        if not args.no_reloc:
            outfile = os.path.join(seq_dir, 'reloc.csv')
            print(outfile)
            with open(outfile, 'w') as f:
                f.write(','.join(csv_header) + '\n')
                for reloc_file_prefix in reloc_file_prefixes:
                    for tag in file_tags:
                        reloc_file = '{}-{}.mat'.format(reloc_file_prefix, tag)
                        line = entry_from_file(seq_dir, reloc_file)
                        f.write(line + '\n')

                    gt_seq_file = '{}-vo-rgb.mat'.format(
                        reloc_file_prefix.split('-')[0])
                    tm_gt = TrajectoryMetrics.loadmat(
                        os.path.join(seq_dir, gt_seq_file))
                    make_plots(seq_dir, reloc_file_prefix,
                               segs, topdown_plane, tm_gt.Twv_gt)


# Do the thing
main()
