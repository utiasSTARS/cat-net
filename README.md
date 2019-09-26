# CAT-Net: Learning Canonical Appearance Transformations

<img src="https://raw.githubusercontent.com/utiasSTARS/cat-net/master/pipeline.png" width="300px"/>

Code to accompany our paper ["How to Train a CAT: Learning Canonical Appearance Transformations for Direct Visual Localization Under Illumination Change"](https://arxiv.org/abs/1709.03009).

## Dependencies
- numpy
- matpotlib
- pytorch + torchvision (1.2)
- Pillow
- progress (for progress bars in train/val/test loops)
- tensorboard + tensorboardX (for visualization)
- [pyslam](https://github.com/utiasSTARS/pyslam) + [liegroups](https://github.com/utiasSTARS/liegroups) (optional, for running odometry/localization experiments)
- OpenCV (optional, for running odometry/localization experiments)

## Training the CAT
1. Download the ETHL dataset from [here](http://cvg.ethz.ch/research/illumination-change-robust-dslam/) 
    1. Rename `ethl1/2` to `ethl1/2_static`.
    2. Update the local paths in `tools/make_ethl_real_sync.py` and run `python3 tools/make_ethl_real_sync.py` to generate a synchronized copy of the `real` sequences.
2. Update the local paths in `run_cat_ethl.py` and run `python3 run_cat_ethl.py` to start training.
3. In another terminal run `tensorboard --port [port] --logdir [path]` to start the visualization server, where `[port]` should be replaced by a numeric value (e.g., 60006) and `[path]` should be replaced by your local results directory.
4. Tune in to `localhost:[port]` and watch the action.

## Running the localization experiments
1. Ensure the [pyslam](https://github.com/utiasSTARS/pyslam) and [liegroups](https://github.com/utiasSTARS/liegroups) packages are installed.
2. Update the local paths in `make_localization_data.py` and run `python3 make_localization_data.py [dataset]` to compile the model outputs into a `localization_data` directory.
3. Update the local paths in `run_localization_[dataset].py` and run `python3 run_localization_[dataset].py`
3. You can compute localization errors against ground truth using the `compute_localization_errors.py` script.

<!-- ## Pre-trained models
Coming soon! -->

## Citation
If you use this code in your research, please cite:
```
@article{2018_Clement_Learning,
  author = {Lee Clement and Jonathan Kelly},
  journal = {{IEEE} Robotics and Automation Letters},
  link = {https://arxiv.org/abs/1709.03009},
  title = {How to Train a {CAT}: Learning Canonical Appearance Transformations for Direct Visual Localization Under Illumination Change},
  year = {2018}
}
```
