import numpy as np

from liegroups import SE3
from pyslam.pipelines import DenseStereoPipeline
from pyslam.sensors import StereoCamera
from pyslam.metrics import TrajectoryMetrics
from pyslam.visualizers import TrajectoryVisualizer
from pyslam.losses import HuberLoss

import time
import os
import argparse

from appearance_transfer.datasets import kitti_affine


def get_camera(test_im, scale):
    # Create the camera
    intrinsics = kitti_affine.intrinsics_centrecrop_256x192
    # intrinsics = kitti_affine.intrinsics_full

    fu = intrinsics.fu * scale
    fv = intrinsics.fv * scale
    cu = intrinsics.cu * scale
    cv = intrinsics.cv * scale
    b = intrinsics.b
    height, width = test_im.shape
    # height = int(height * scale)
    # width = int(width * scale)
    return StereoCamera(cu, cv, fu, fv, b, width, height)


def do_vo_mapping(basepath, sequence, ref_condition,
                  scale=1., frames=None, outfile=None,
                  left_dir='left', right_dir='right'):
    ref_data = kitti_affine.Dataset(basepath, sequence,
                                    ref_condition, frames=frames,
                                    left_dir=left_dir, right_dir=right_dir)

    test_im = ref_data.get_gray(0)[0]
    camera = get_camera(test_im, scale)
    camera.maxdepth = 200.

    # Ground truth
    T_w_c_gt = [SE3.from_matrix(T, normalize=True) for T in ref_data.poses]
    T_0_w = T_w_c_gt[0].inv()

    vo = DenseStereoPipeline(camera, first_pose=T_0_w)
    vo.depth_map_type = 'disparity'
    vo.keyframe_trans_thresh = 1.  # meters
    vo.keyframe_rot_thresh = 10. * np.pi / 180.  # rad
    vo.depth_stiffness = 1. / 0.1  # 1/pixels (disparity in this case)
    vo.intensity_stiffness = 1. / 0.3  # 1/ (intensity is in [0,1] )
    vo.use_motion_model_guess = False
    # vo.min_grad = 0.2
    # vo.loss = HuberLoss(5.0)

    print('Mapping using {}/{}'.format(sequence, ref_condition))
    vo.set_mode('map')

    start = time.perf_counter()
    for c_idx in range(len(ref_data)):
        im_left, im_right = ref_data.get_gray(c_idx)

        vo.track(im_left, im_right)
        # vo.track(im_left, im_right, guess=T_w_c_gt[c_idx].inv())
        end = time.perf_counter()
        print('Image {}/{} ({:.2f} %) | {:.3f} s'.format(
            c_idx, len(ref_data), 100. * c_idx / len(ref_data), end - start), end='\r')
        start = end

    # Compute errors
    T_w_c_est = [T.inv() for T in vo.T_c_w]
    tm = TrajectoryMetrics(T_w_c_gt, T_w_c_est)

    # Save to file
    if outfile is not None:
        print('Saving to {}'.format(outfile))
        tm.savemat(outfile)

    return tm, vo


def do_tracking(basepath, sequence, track_condition, vo,
                scale=1., frames=None, outfile=None,
                left_dir='left', right_dir='right'):
    track_data = kitti_affine.Dataset(basepath, sequence,
                                      track_condition, frames=frames,
                                      left_dir=left_dir, right_dir=right_dir)

    # Ground truth
    T_w_c_gt = [SE3.from_matrix(T, normalize=True) for T in track_data.poses]
    T_0_w = T_w_c_gt[0].inv()

    print('Tracking using {}/{}'.format(sequence, track_condition))
    vo.set_mode('track')

    start = time.perf_counter()
    for c_idx in range(len(track_data)):
        im_left, im_right = track_data.get_gray(c_idx)

        try:
            vo.track(im_left, im_right)
            # vo.track(im_left, im_right, guess=T_w_c_gt[c_idx].inv())
            end = time.perf_counter()
            print('Image {}/{} ({:.2f} %) | {:.3f} s'.format(
                c_idx, len(track_data), 100. * c_idx / len(track_data), end - start), end='\r')
            start = end

        except Exception as e:
            print('Error on {}/{}'.format(sequence, track_condition))
            print(e)
            print('Image {}/{} ({:.2f} %) | {:.3f} s'.format(
                c_idx, len(track_data), 100. * c_idx / len(track_data), end - start))
            break

        start = end

    # Compute errors
    T_w_c_est = [T.inv() for T in vo.T_c_w]
    tm = TrajectoryMetrics(T_w_c_gt, T_w_c_est)

    # Save to file
    if outfile is not None:
        print('Saving to {}'.format(outfile))
        tm.savemat(outfile)

    return tm, vo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--use_cat', help='use canonical appearance transformation', action='store_true')
    parser.add_argument(
        '--no_vo', help='skip VO experiments', action='store_true')
    parser.add_argument(
        '--no_reloc', help='skip VO experiments', action='store_true')
    args = parser.parse_args()

    if args.use_cat:
        left_dir = 'left-cat'
        right_dir = 'right-cat'
    else:
        left_dir = 'left'
        right_dir = 'right'

    datadir = '/media/raid5-array/experiments/appearance-transfer/virtual-kitti/vkitti-to-kitti/localization-data'
    outdir = '/media/raid5-array/experiments/appearance-transfer/virtual-kitti/vkitti-to-kitti/pyslam'
    os.makedirs(outdir, exist_ok=True)

    seqs = ['0018']
    vo_conditions = ['clone', 'light', 'dark']
    reloc_conditions = {'clone': ['clone', 'light', 'dark']}

    for seq in seqs:
        # Do VO
        if not args.no_vo:
            for cond in vo_conditions:
                print('Doing VO on {}/{}'.format(seq, cond))

                if args.use_cat:
                    outfile = os.path.join(
                        outdir, seq + '-' + cond + '-vo-cat.mat')
                else:
                    outfile = os.path.join(
                        outdir, seq + '-' + cond + '-vo.mat')

                tm, vo = do_vo_mapping(datadir, seq, cond,
                                       outfile=outfile, left_dir=left_dir, right_dir=right_dir)

        # Do relocalization
        if not args.no_reloc:
            for ref_cond, track_conds in reloc_conditions.items():
                _, vo = do_vo_mapping(datadir, seq, ref_cond,
                                      left_dir=left_dir, right_dir=right_dir)

                for track_cond in track_conds:
                    print('Sequence {} | Reference condition {} | Tracking condition {}'.format(
                        seq, ref_cond, track_cond))

                    if args.use_cat:
                        outfile = os.path.join(
                            outdir, seq + '-' + ref_cond +
                            '-' + track_cond + '-cat.mat')
                    else:
                        outfile = os.path.join(
                            outdir, seq + '-' + ref_cond +
                            '-' + track_cond + '.mat')

                    tm, _ = do_tracking(
                        datadir, seq, track_cond, vo, outfile=outfile, left_dir=left_dir, right_dir=right_dir)


# Do the thing
main()
