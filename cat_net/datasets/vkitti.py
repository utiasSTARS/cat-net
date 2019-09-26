import torch.utils.data
from torchvision import transforms

import os.path
import glob
import numpy as np

from collections import namedtuple
from PIL import Image

from .. import transforms as custom_transforms

CameraIntrinsics = namedtuple('CameraIntrinsics', 'fu, fv, cu, cv')
intrinsics_full = CameraIntrinsics(725.0, 725.0, 620.5, 187.0)
# 1242x375 --> 256x192
intrinsics_centrecrop_256x192 = CameraIntrinsics(371.2, 371.2, 127.5, 95.5)


class Dataset:
    """Load and parse data from Virtual KITTI dataset."""

    def __init__(self, base_path, sequence, condition, **kwargs):
        self.base_path = base_path
        self.sequence = sequence
        self.condition = condition
        self.frames = kwargs.get('frames', None)
        self.rgb_dir = kwargs.get('rgb_dir', 'vkitti_1.3.1_rgb')
        self.depth_dir = kwargs.get('depth_dir', 'vkitti_1.3.1_depthgt')
        self.gt_dir = kwargs.get('gt_dir', 'vkitti_1.3.1_extrinsicsgt')
        self.pose_file = kwargs.get('pose_file',
                                    os.path.join(self.base_path, self.gt_dir,
                                                 '{}_{}.txt'.format(
                                                     self.sequence, self.condition)))

        self._load_timestamps_and_poses()

        self.num_frames = len(self.timestamps)

        self.rgb_files = sorted(glob.glob(
            os.path.join(self.base_path, self.rgb_dir,
                         self.sequence, self.condition, '*.png')))
        self.depth_files = sorted(glob.glob(
            os.path.join(self.base_path, self.depth_dir,
                         self.sequence, self.condition, '*.png')))

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
                                mode='F', dtype=np.float, factor=100.)

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
                if line[0] == 'frame':  # this is the header
                    continue
                self.timestamps.append(float(line[0]))

                # from world to camera
                Tmatrix = np.array([float(x)
                                    for x in line[1:17]]).reshape((4, 4))
                # from camera to world
                self.poses.append(np.linalg.inv(Tmatrix))

        if self.frames is not None:
            self.timestamps = [self.timestamps[i] for i in self.frames]
            self.poses = [self.poses[i] for i in self.frames]


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
