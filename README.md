# CCoopNets

This repo contains Tensorflow implementation for the paper:

[Cooperative Training of Fast Thinking Initializer and Slow Thinking Solver for Conditional Learning](http://www.stat.ucla.edu/~jxie/CCoopNets/CCoopNets_file/doc/CCoopNets.pdf)  
In TPAMI 2021

[**Paper**](http://www.stat.ucla.edu/~jxie/CCoopNets/CCoopNets_file/doc/CCoopNets.pdf) | [**Project**](http://www.stat.ucla.edu/~jxie/CCoopNets/)

## Getting Started
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

### Exp1: Attribute to Image
- Train [mnist](http://yann.lecun.com/exdb/mnist/) data
```
sh scripts/train_cat2img_mnist.sh --dataroot datasets
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
