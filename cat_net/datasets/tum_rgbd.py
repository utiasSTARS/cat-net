import torch.utils.data
from torchvision import transforms

import os.path
import glob
import numpy as np

from collections import namedtuple
from PIL import Image

from .. import transforms as custom_transforms

CameraIntrinsics = namedtuple('CameraIntrinsics', 'fu, fv, cu, cv')
# https://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html
ethl1_intrinsics = CameraIntrinsics(481.20, -480.00, 319.50, 239.50)  # 640x480
ethl1_intrinsics_256x192 = CameraIntrinsics(192.48, -192.00, 127.80, 95.80)
ethl2_intrinsics = ethl1_intrinsics  # 640x480
ethl2_intrinsics_256x192 = ethl1_intrinsics_256x192
# http://cvg.ethz.ch/research/illumination-change-robust-dslam/
real_intrinsics = CameraIntrinsics(538.7, 540.7, 319.2, 233.6)  # 640x480
real_intrinsics_256x192 = CameraIntrinsics(215.48, 216.28, 127.68, 93.44)


class Dataset:
    """Load and parse data in TUM RGB-D format.

       Reference: http://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
    """

    def __init__(self, base_path, sequence, condition, **kwargs):
        self.base_path = base_path
        self.sequence = sequence
        self.condition = condition

        self.data_path = os.path.join(
            self.base_path, self.sequence + '_' + self.condition)

        self.frames = kwargs.get('frames', None)
        self.rgb_dir = kwargs.get('rgb_dir', 'rgb')
        self.depth_dir = kwargs.get('depth_dir', 'depth')
        self.pose_file = kwargs.get('pose_file', os.path.join(
            self.data_path, 'groundtruth.txt'))

        self._load_timestamps_and_poses()

        self.num_frames = len(self.timestamps)

        self.rgb_files = sorted(glob.glob(
            os.path.join(self.data_path, self.rgb_dir, '*.png')))
        self.depth_files = sorted(glob.glob(
            os.path.join(self.data_path, self.depth_dir, '*.png')))

        if self.frames is not None:
            self.rgb_files = [self.rgb_files[i] for i in self.frames]
            self.depth_files = [self.depth_files[i] for i in self.frames]

    def __len__(self):
        return self.num_frames

    def get_rgb(self, idx, size=None):
        """Load RGB image from file."""
        return self._load_image(self.rgb_files[idx], size, mode='RGB', dtype=np.uint8)

    def get_gray(self, idx, size=None):
        """Load grayscale image from file."""
        return self._load_image(self.rgb_files[idx], size, mode='L', dtype=np.uint8)

    def get_depth(self, idx, size=None):
        """Load depth image from file."""
        return self._load_image(self.depth_files[idx], size,
                                mode='F', dtype=np.float, factor=5000)

    def _load_image(self, impath, size=None, mode='RGB', dtype=np.float, factor=1):
        """Load image from file."""
        im = Image.open(impath).convert(mode)
        if size:
            im = im.resize(size, resample=Image.BILINEAR)
        return (np.array(im) / factor).astype(dtype)

    def _load_timestamps_and_poses(self):
        """Load ground truth poses (T_w_cam) and timestamps from file."""
        self.timestamps = []
        self.poses = []

        # Read and parse the poses
        with open(self.pose_file, 'r') as f:
            for line in f.readlines():
                line = line.split()
                if line[0] is '#':  # this is a comment
                    continue
                self.timestamps.append(float(line[0]))

                t = np.array([float(x) for x in line[1:4]])
                q = np.array([float(x) for x in line[4:]])
                q = q / np.linalg.norm(q)
                self.poses.append((q, t))  # from camera to world

        if self.frames is not None:
            self.timestamps = [self.timestamps[i] for i in self.frames]
            self.poses = [self.poses[i] for i in self.frames]


class LocalizationDataset(Dataset):
    def __init__(self, base_path, sequence, condition, **kwargs):
        self.base_path = base_path
        self.sequence = sequence
        self.condition = condition
        self.frames = kwargs.get('frames', None)
        self.rgb_dir = kwargs.get('rgb_dir', 'rgb')
        self.depth_dir = kwargs.get('depth_dir', 'depth')
        self.gt_file = kwargs.get('gt_file', 'groundtruth.txt')

        self.data_path = os.path.join(
            self.base_path, self.sequence, self.condition)

        self.pose_file = os.path.join(self.data_path, self.gt_file)
        self._load_timestamps_and_poses()

        self.num_frames = len(self.timestamps)

        self.rgb_files = sorted(glob.glob(
            os.path.join(self.base_path, self.sequence,
                         self.condition, self.rgb_dir, '*.png')))
        self.depth_files = sorted(glob.glob(
            os.path.join(self.base_path, self.sequence,
                         self.condition, self.depth_dir, '*.png')))

        if self.frames is not None:
            self.rgb_files = [self.rgb_files[i] for i in self.frames]
            self.depth_files = [self.depth_files[i] for i in self.frames]


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, opts, sequence, source_condition, target_condition, random_crop, **kwargs):
        self.opts = opts
        self.random_crop = random_crop

        self.source = Dataset(self.opts.data_dir, sequence,
                              source_condition, **kwargs)
        self.target = Dataset(self.opts.data_dir, sequence,
                              target_condition, **kwargs)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        source = Image.fromarray(self.source.get_rgb(idx))
        target = Image.fromarray(self.target.get_rgb(idx))

        transform = transforms.Compose([
            transforms.Resize(min(self.opts.image_load_size)),
            transforms.CenterCrop(self.opts.image_load_size),
            custom_transforms.StatefulRandomCrop(
                self.opts.image_final_size) if self.random_crop else transforms.Resize(self.opts.image_final_size),
            transforms.ToTensor(),
            transforms.Normalize(self.opts.image_mean, self.opts.image_std)
        ])

        source = transform(source)
        target = transform(target)

        data = {'source': source,
                'target': target}

        return data
