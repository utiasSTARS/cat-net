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

## Running the demo experiment
1. Download the ETHL dataset from [here](http://cvg.ethz.ch/research/illumination-change-robust-dslam/).
2. Update `run_cat_experiment.py` to point to the appropriate local paths.
3. In a terminal run `python3 -m visdom.server -port 8097` to start the visualization server.
4. In another terminal run `python3 run_cat_experiment.py` to start training.
5. Tune in to `localhost:8097` and watch the fun.

## Running the localization experiments
*Note: the scripts referenced here are from an older version of the repository and may need some adjustments.*
1. Ensure the [pyslam](https://github.com/utiasSTARS/pyslam) and [liegroups](https://github.com/utiasSTARS/liegroups) packages are installed
2. In a terminal open the `localization` directory and run `python3 run_localization_[dataset].py`
3. You can compute localization errors against ground truth using the `compute_localization_errors.py` script.

## Pre-trained models
Coming soon!

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
