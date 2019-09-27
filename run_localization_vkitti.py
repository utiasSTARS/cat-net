import numpy as np
import cv2

from liegroups import SE3
from pyslam.pipelines import DenseRGBDPipeline
from pyslam.sensors import RGBDCamera
from pyslam.metrics import TrajectoryMetrics
from pyslam.visualizers import TrajectoryVisualizer

import time
import os
import argparse

from cat_net.datasets import vkitti


def get_camera(seq_name, test_im):
    # Create the camera
    intrinsics = vkitti.intrinsics_centrecrop_256x192

    fu = intrinsics.fu
    fv = intrinsics.fv
    cu = intrinsics.cu
    cv = intrinsics.cv
    height, width = test_im.shape
    return RGBDCamera(cu, cv, fu, fv, width, height)


def do_vo_mapping(basepath, seq, ref_cond, frames=None, outfile=None, rgb_dir='rgb'):
    ref_data = vkitti.LocalizationDataset(
        basepath, seq, ref_cond, frames=frames, rgb_dir=rgb_dir)

    test_im = ref_data.get_gray(0)
    camera = get_camera(seq, test_im)
    camera.maxdepth = 200.

    # Ground truth
    T_w_c_gt = [SE3.from_matrix(p, normalize=True) for p in ref_data.poses]
    T_0_w = T_w_c_gt[0].inv()

    vo = DenseRGBDPipeline(camera, first_pose=T_0_w)
    # vo.keyframe_trans_thresh = 3.  # meters
    vo.keyframe_trans_thresh = 2.  # meters
    vo.keyframe_rot_thresh = 15. * np.pi / 180.  # rad
    vo.depth_stiffness = 1. / 0.01  # 1/meters
    vo.intensity_stiffness = 1. / 0.2  # 1/ (intensity is in [0,1] )
    # vo.intensity_stiffness = 1. / 0.1
    vo.use_motion_model_guess = True
    # vo.min_grad = 0.2
    # vo.loss = HuberLoss(5.0)

    print('Mapping using {}/{}'.format(seq, ref_cond))
    vo.set_mode('map')

    start = time.perf_counter()
    for c_idx in range(len(ref_data)):
        image = ref_data.get_gray(c_idx)
        depth = ref_data.get_depth(c_idx)

        depth[depth >= camera.maxdepth] = -1.
        vo.track(image, depth)
        # vo.track(image, depth, guess=T_w_c_gt[c_idx].inv()) # debug
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


def do_tracking(basepath, seq, track_cond, vo, frames=None, outfile=None, rgb_dir='rgb'):
    track_data = vkitti.LocalizationDataset(
        basepath, seq, track_cond, frames=frames, rgb_dir=rgb_dir)

    # Ground truth
    T_w_c_gt = [SE3.from_matrix(p, normalize=True) for p in track_data.poses]
    T_0_w = T_w_c_gt[0].inv()

    print('Tracking using {}/{}'.format(seq, track_cond))
    vo.set_mode('track')

    start = time.perf_counter()
    for c_idx in range(len(track_data)):
        image = track_data.get_gray(c_idx)
        depth = track_data.get_depth(c_idx)
        try:
            depth[depth >= vo.camera.maxdepth] = -1.
            vo.track(image, depth)
            # vo.track(image, depth, guess=T_w_c_gt[c_idx].inv()) # debug
            end = time.perf_counter()
            print('Image {}/{} ({:.2f} %) | {:.3f} s'.format(
                c_idx, len(track_data), 100. * c_idx / len(track_data), end - start), end='\r')

        except Exception as e:
            print('Error on {}/{}'.format(seq, track_cond))
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
        'image', help='which rgb dir', type=str, choices=['rgb', 'cat'])
    parser.add_argument(
        '--no_vo', help='skip VO experiments', action='store_true')
    parser.add_argument(
        '--no_reloc', help='skip VO experiments', action='store_true')
    args = parser.parse_args()

    rgb_dir = args.image if args.image == 'rgb' else 'rgb_' + args.image

    basedir = '/media/raid5-array/experiments/cat-net/virtual-kitti'
    datadir = os.path.join(basedir, 'localization_data')
    outdir = os.path.join(basedir, 'pyslam')
    os.makedirs(outdir, exist_ok=True)

    seqs = ['0001', '0002', '0006', '0018', '0020']
    conds = ['clone', 'morning', 'overcast', 'sunset']
    canonical = conds[2]

    # Do VO
    if not args.no_vo:
        for seq in seqs:
            for cond in conds:
                print('Doing VO on {}/{}'.format(seq, cond))

                seq_outdir = os.path.join(outdir, seq)
                os.makedirs(seq_outdir, exist_ok=True)
                outfile = os.path.join(
                    seq_outdir, '{}-vo-{}.mat'.format(cond, args.image))

                tm, vo = do_vo_mapping(datadir, seq, cond,
                                       outfile=outfile, rgb_dir=rgb_dir)

    # Do relocalization
    if not args.no_reloc:
        for seq in seqs:
            # Map in the canonical condition
            _, vo = do_vo_mapping(datadir, seq, canonical, rgb_dir=rgb_dir)

        for cond in conds:
            print('Seq: {} | Ref: {} | Track: {}'.format(seq, canonical, cond))

            seq_outdir = os.path.join(outdir, seq)
            os.makedirs(seq_outdir, exist_ok=True)
            outfile = os.path.join(
                seq_outdir, '{}-{}-{}.mat'.format(canonical, cond, args.image))

            tm, _ = do_tracking(datadir, seq, cond, vo,
                                outfile=outfile, rgb_dir=rgb_dir)


# Do the thing
main()
