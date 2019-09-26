import os
import shutil
import re
import numpy as np

dataset_dir = '/media/m2-drive/datasets/ethl_dataset/raw'

datasets = [d for d in os.listdir(dataset_dir) if 'real_' in d]

for input_dataset in datasets:
    tokens = input_dataset.split('_')
    output_dataset = tokens[0] + '_sync_' + tokens[1]

    input_dir = os.path.join(dataset_dir, input_dataset)
    input_rgb_dir = os.path.join(input_dir, 'rgb')
    input_depth_dir = os.path.join(input_dir, 'depth')
    input_associations_file = os.path.join(input_dir, 'associations.txt')
    input_groundtruth_file = os.path.join(input_dir, 'groundtruth.txt')

    output_dir = os.path.join(dataset_dir, output_dataset)
    output_rgb_dir = os.path.join(output_dir, 'rgb')
    output_depth_dir = os.path.join(output_dir, 'depth')
    output_associations_file = os.path.join(output_dir, 'associations.txt')
    output_groundtruth_file = os.path.join(output_dir, 'groundtruth.txt')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_rgb_dir, exist_ok=True)
    os.makedirs(output_depth_dir, exist_ok=True)

    # Get correspondences
    association_list = []
    print('reading associations from {}'.format(input_associations_file))
    with open(input_associations_file, 'r') as f:
        for line in f.readlines():
            # timestamp_rgb, filename_rgb, timestamp_depth, filename_depth
            association_list.append(re.split('\s', line))

    timestamps_rgb = [assoc[0] for assoc in association_list]
    img_list_rgb = [assoc[1] for assoc in association_list]
    img_list_depth = [assoc[3] for assoc in association_list]

    # Update groundtruth as well
    groundtruth_list = []
    print('reading groundtruth from {}'.format(input_groundtruth_file))
    with open(input_groundtruth_file, 'r') as f:
        for line in f.readlines():
            # timestamp, [pose]
            groundtruth_list.append([float(d)
                                     for d in re.split('\s', line) if len(d) > 0])

    timestamps_gt = np.array([gt[0] for gt in groundtruth_list])
    assoc_gt_list = []
    for rgb_time in timestamps_rgb:
        abs_time_diffs = np.abs(timestamps_gt - float(rgb_time))
        closest_idx = np.argmin(abs_time_diffs)
        assoc_gt_list.append(groundtruth_list[closest_idx])

    print('writing groundtruth to {}'.format(output_groundtruth_file))
    with open(output_groundtruth_file, 'w') as f:
        gt_strings = []
        for gt in assoc_gt_list:
            gt_as_strings = [str(x) for x in gt]
            gt_strings.append(' '.join(gt_as_strings) + '\n')

        f.writelines(gt_strings)

    # Copy rgb files
    for f in img_list_rgb:
        inpath = os.path.join(input_rgb_dir, f)
        outpath = os.path.join(output_rgb_dir, f)
        print('copying {} -> {}'.format(inpath, outpath))
        shutil.copy(inpath, outpath)

    # Copy depth files
    for f in img_list_depth:
        inpath = os.path.join(input_depth_dir, f)
        outpath = os.path.join(output_depth_dir, f)
        print('copying {} -> {}'.format(inpath, outpath))
        shutil.copy(inpath, outpath)
