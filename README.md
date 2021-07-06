# CCoopNets

This repo contains Tensorflow implementation for the paper:

[Cooperative Training of Fast Thinking Initializer and Slow Thinking Solver for Conditional Learning](http://www.stat.ucla.edu/~jxie/CCoopNets/CCoopNets_file/doc/CCoopNets.pdf)  
In TPAMI 2021

[**Paper**](http://www.stat.ucla.edu/~jxie/CCoopNets/CCoopNets_file/doc/CCoopNets.pdf) | [**Project**](http://www.stat.ucla.edu/~jxie/CCoopNets/)

## Getting Started
### Install
- Dependencies: 
    - Python3
    - Tensorflow1.4+

- Clone this repo:
```
git clone https://github.com/zilongzheng/CCoopNets.git
```
- Install python requirements:
```
pip install -r requirements.txt
```

### Datasets
We use datasets from the following resources. Download before you are using any of these.
- [cityscapes](https://www.cityscapes-dataset.com/)
- [CMP Facade](https://cmp.felk.cvut.cz/~tylecr1/facade/)
  - You can create dataset by
  ```bash
  bash ./scripts/prepro/make_facades_dataset.sh [base|extended]
  ```
- [CUHK Face](http://mmlab.ie.cuhk.edu.hk/archive/facesketch.html)
- [UT-Zap50K](http://vision.cs.utexas.edu/projects/finegrained/utzap50k/)
  - The preprocessing code is [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master/scripts/edges).

**Note:** We use ``dataroot/category/<original datapath>`` for most of the datasets, change relative path declared [here](./scripts/train_img2img.py) based on your path organization.


### Exp1: Category to Image
- Train [mnist](http://yann.lecun.com/exdb/mnist/) data
```
sh scripts/train_cat2img_mnist.sh --dataroot datasets --output_dir output
```

### Exp2: Image to Image
- Train facade2photo
```
sh scripts/train_img2img_facade2photo.sh --dataroot datasets --output_dir output
```

### Logs
The output is recorded using [tensorboard](https://www.tensorflow.org/tensorboard/), which can be visualized by
```
tensorboard --logdir output/<category>_<timestamp>/log/
```


## Citation
If you use this code for your research, please cite our paper.
```bibtex
@article{xie2021ccoopnets,
  author={Xie, Jianwen and Zheng, Zilong and Fang, Xiaolin and Zhu, Song-Chun and Wu, Ying Nian},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Cooperative Training of Fast Thinking Initializer and Slow Thinking Solver for Conditional Learning}, 
  year={2021},
  doi={10.1109/TPAMI.2021.3069023}}
```
