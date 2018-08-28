import numpy as np
import cv2

from liegroups import SE3
from pyslam.pipelines import DenseRGBDPipeline
from pyslam.sensors import RGBDCamera
from pyslam.metrics import TrajectoryMetrics
from pyslam.visualizers import TrajectoryVisualizer
from pyslam.losses import HuberLoss

import time
import os
import argparse

import tools.vkitti as vkitti


def get_camera(seq_name, test_im, scale):
    # Create the camera
    intrinsics = vkitti.intrinsics_centrecrop_256x192
    # intrinsics = vkitti.intrinsics_full

    fu = intrinsics.fu * scale
    fv = intrinsics.fv * scale
    cu = intrinsics.cu * scale
    cv = intrinsics.cv * scale
    height, width = test_im.shape
    # height = int(height * scale)
    # width = int(width * scale)
    return RGBDCamera(cu, cv, fu, fv, width, height)


def do_vo_mapping(basepath, ref_seq, scale=1., frames=None, outfile=None, rgb_dir='rgb'):
    ref_data = vkitti.Dataset(
        basepath, ref_seq, frames=frames, rgb_dir=rgb_dir)

    test_im = next(ref_data.gray)
    camera = get_camera(ref_seq, test_im, scale)
    camera.maxdepth = 200.

    # Ground truth
    T_w_c_gt = ref_data.poses
    T_0_w = T_w_c_gt[0].inv()

    vo = DenseRGBDPipeline(camera, first_pose=T_0_w)
    vo.keyframe_trans_thresh = 3.  # meters
    vo.keyframe_rot_thresh = 15. * np.pi / 180.  # rad
    vo.depth_stiffness = 1. / 0.01  # 1/meters
    vo.intensity_stiffness = 1. / 0.2  # 1/ (intensity is in [0,1] )
    vo.use_motion_model_guess = True
    # vo.min_grad = 0.2
    # vo.loss = HuberLoss(5.0)

    print('Mapping using {}'.format(ref_seq))
    vo.set_mode('map')

    start = time.perf_counter()
    for c_idx, (image, depth) in enumerate(zip(ref_data.gray, ref_data.depth)):
        depth[depth >= camera.maxdepth] = -1.
        vo.track(image, depth)
        # vo.track(image, depth, guess=T_w_c_gt[c_idx].inv())
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


def do_tracking(basepath, track_seq, vo, scale=1., frames=None, outfile=None, rgb_dir='rgb'):
    track_data = vkitti.Dataset(
        basepath, track_seq, frames=frames, rgb_dir=rgb_dir)

    # Ground truth
    T_w_c_gt = track_data.poses
    T_0_w = T_w_c_gt[0].inv()

    print('Tracking using {}'.format(track_seq))
    vo.set_mode('track')

    start = time.perf_counter()
    for c_idx, (image, depth) in enumerate(zip(track_data.gray, track_data.depth)):
        try:
            depth[depth >= vo.camera.maxdepth] = -1.
            vo.track(image, depth)
            # vo.track(image, depth, guess=T_w_c_gt[c_idx].inv())
            end = time.perf_counter()
            print('Image {}/{} ({:.2f} %) | {:.3f} s'.format(
                c_idx, len(track_data), 100. * c_idx / len(track_data), end - start), end='\r')

        except Exception as e:
            print('Error on {}'.format(track_seq))
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
        rgb_dir = 'rgb_cat'
    else:
        rgb_dir = 'rgb'

    datadir = '/Users/leeclement/Desktop/image-corrector-net/virtual-kitti/localization_data'
    outdir = '/Users/leeclement/Desktop/image-corrector-net/pyslam/virtual-kitti'
    os.makedirs(outdir, exist_ok=True)

    vo_seqs = ['0001_overcast', '0001_clone', '0001_morning', '0001_sunset',
               '0002_overcast', '0002_clone', '0002_morning', '0002_sunset',
               '0006_overcast', '0006_clone', '0006_morning', '0006_sunset',
               '0018_overcast', '0018_clone', '0018_morning', '0018_sunset',
               '0020_overcast', '0020_clone', '0020_morning', '0020_sunset']

    reloc_seqs = {'0001_overcast': ['0001_overcast', '0001_clone', '0001_morning', '0001_sunset'],
                  '0002_overcast': ['0002_overcast', '0002_clone', '0002_morning', '0002_sunset'],
                  '0006_overcast': ['0006_overcast', '0006_clone', '0006_morning', '0006_sunset'],
                  '0018_overcast': ['0018_overcast', '0018_clone', '0018_morning', '0018_sunset'],
                  '0020_overcast': ['0020_overcast', '0020_clone', '0020_morning', '0020_sunset']}

    # Do VO
    if not args.no_vo:
        for seq in vo_seqs:
            # if '0018' not in seq:
            #     continue

            print('Doing VO on {}'.format(seq))

            if args.use_cat:
                outfile = os.path.join(outdir, seq + '-vo-cat.mat')
            else:
                outfile = os.path.join(outdir, seq + '-vo.mat')

            tm, vo = do_vo_mapping(datadir, seq,
                                   outfile=outfile, rgb_dir=rgb_dir)

    # Do relocalization
    if not args.no_reloc:
        for ref_seq, track_seqs in reloc_seqs.items():
            # if '0018' not in ref_seq:
            #     continue
            # Don't use CAT on map imagery for relocalization experiments
            # _, vo = do_vo_mapping(datadir, ref_seq, rgb_dir='rgb')
            # Or do...
            _, vo = do_vo_mapping(datadir, ref_seq, rgb_dir=rgb_dir)

            for track_seq in track_seqs:
                print('Reference sequence {} | Tracking sequence {}'.format(
                    ref_seq, track_seq))

                if args.use_cat:
                    outfile = os.path.join(
                        outdir, ref_seq + '-' + track_seq + '-cat.mat')
                else:
                    outfile = os.path.join(
                        outdir, ref_seq + '-' + track_seq + '.mat')

                tm, _ = do_tracking(datadir, track_seq, vo,
                                    outfile=outfile, rgb_dir=rgb_dir)


# Do the thing
main()
