# CAT-Net: Learning Canonical Appearance Transformations

<img src="https://raw.githubusercontent.com/utiasSTARS/cat-net/master/pipeline.png" width="300px"/>

Code to accompany our paper ["How to Train a CAT: Learning Canonical Appearance Transformations for Robust Direct Visual Localization Under Illumination Change"](https://arxiv.org/abs/1709.03009).

Dependencies: 
- numpy
- pytorch + torchvision (0.3.0)
- PIL
- visdom

## Running the demo experiment
1. Download the ETHL dataset from [here](http://cvg.ethz.ch/research/illumination-change-robust-dslam/).
2. Update `run_cat_experiment.py` to point to the appropriate local paths.
3. In a terminal run `python3 -m visdom.server -port 8097` to start the visualization server.
4. In another terminal run `python3 run_cat_experiment.py` to start training.
5. Tune in to `localhost:8097` and watch the fun.

## Citation
If you use this code in your research, please cite:
```
@article{2018_Clement_Learning,
  author = {Lee Clement and Jonathan Kelly},
  link = {https://arxiv.org/abs/1709.03009},
  title = {How to Train a {CAT}: Learning Canonical Appearance Transformations for Robust Direct Localization Under Illumination Change},
  year = {2018}
}
```