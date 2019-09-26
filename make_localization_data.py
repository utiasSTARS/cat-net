import os
import glob
import shutil
import argparse

from PIL import Image
from torchvision import transforms

from cat_net.datasets import vkitti, tum_rgbd
from cat_net.options import Options

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, choices=[
                    'ethl_dataset', 'virtual-kitti'])
args = parser.parse_args()

opts = Options()

datasets_dir = '/media/m2-drive/datasets/'
experiments_dir = '/media/raid5-array/experiments/cat-net/'

results_dir = os.path.join(experiments_dir, args.dataset)
localization_data_dir = os.path.join(
    experiments_dir, 'localization_data', args.dataset)

os.makedirs(localization_data_dir, exist_ok=True)

seq_dirs = [d for d in os.listdir(
    results_dir) if '-test' in d]

for seq_dir in seq_dirs:
    seq = seq_dir.split('-')[0]
    cond_dirs = [d for d in os.listdir(
        os.path.join(results_dir, seq_dir)) if '-test' in d]

    for cond_dir in cond_dirs:
        cond = cond_dir.split('-')[0]

        if args.dataset == 'ethl_dataset':
            data_dir = os.path.join(datasets_dir, args.dataset, 'raw')
            data = tum_rgbd.Dataset(data_dir, seq, cond)
        elif args.dataset == 'virtual-kitti':
            data_dir = os.path.join(datasets_dir, args.dataset, 'raw')
            data = vkitti.Dataset(data_dir, seq, cond)

        in_rgb_dir = os.path.join(
            results_dir, seq_dir, cond_dir, 'test_best', 'source')
        in_cat_dir = os.path.join(
            results_dir, seq_dir, cond_dir, 'test_best', 'output')

        out_dir = os.path.join(localization_data_dir, seq, cond)
        out_rgb_dir = os.path.join(out_dir, 'rgb')
        out_cat_dir = os.path.join(out_dir, 'rgb_cat')
        out_depth_dir = os.path.join(out_dir, 'depth')
        out_pose_file = os.path.join(out_dir, 'groundtruth.txt')

        os.makedirs(out_rgb_dir, exist_ok=True)
        os.makedirs(out_cat_dir, exist_ok=True)
        os.makedirs(out_depth_dir, exist_ok=True)

        in_rgb_files = sorted(glob.glob(os.path.join(in_rgb_dir, '*.png')))
        in_cat_files = sorted(glob.glob(os.path.join(in_cat_dir, '*.png')))
        in_depth_files = data.depth_files

        out_rgb_files = [os.path.join(
            out_rgb_dir, f.split('/')[-1]) for f in in_rgb_files]
        out_cat_files = [os.path.join(
            out_cat_dir, f.split('/')[-1]) for f in in_cat_files]
        out_depth_files = [os.path.join(
            out_depth_dir, f.split('/')[-1]) for f in in_rgb_files]

        print('copying images from {} -> {}'.format(in_rgb_dir, out_rgb_dir))
        for in_file, out_file in zip(in_rgb_files, out_rgb_files):
            shutil.copyfile(in_file, out_file)

        print('copying images from {} -> {}'.format(in_cat_dir, out_cat_dir))
        for in_file, out_file in zip(in_cat_files, out_cat_files):
            shutil.copyfile(in_file, out_file)

        transform = transforms.Compose([
            transforms.Resize(min(opts.image_load_size)),
            transforms.CenterCrop(opts.image_load_size),
            transforms.Resize(opts.image_final_size)])

        print('creating resized depth images in {}'.format(out_depth_dir))
        for in_file, outfile in zip(in_depth_files, out_depth_files):
            depth = Image.open(in_file)
            depth = transform(depth)
            depth.save(outfile)

        print('copying ground truth from {} -> {}'.format(data.pose_file, out_pose_file))
        shutil.copyfile(data.pose_file, out_pose_file)
