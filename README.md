## Torwards Gradient-based Bilevel Optimization with non-convex Followers and Beyond
This repo contains code accompaning the paper, Torwards Gradient-based Bilevel Optimization with non-convex Followers and Beyond (Liu et al., NeurIPS 2021). It includes code for running the numerical example, few-shot classification and data hyper-cleaning experiments.

### Abstract
In recent years, Bi-Level Optimization (BLO) techniques have received extensive attentions from both learning and vision communities. A variety of BLO models in complex and practical tasks are of non-convex follower structure in nature (a.k.a., without Lower-Level Convexity, LLC for short). However, this challenging class of BLOs is lack of developments on both efficient solution strategies and solid the oretical guarantees. In this work, we propose a new algorithmic framework, named Initialization Auxiliary and Pessimistic Trajectory Truncated Gradient Method (IAPTT-GM), to partially address the above issues. In particular, by introducing an auxiliary as initialization to guide the optimization dynamics and designing a pessimistic trajectory truncation operation, we construct a reliable approximate version of the original BLO in the absence of LLC hypothesis. Our theoretical investigations establish the convergence of solutions returned by IAPTT-GM towards those of the original BLO without LLC. As an additional bonus, we also theoretically justify the quality of our IAPTT-GM embedded with Nesterov’s accelerated dynamics under LLC. The experimental results confirm both the convergence of our algorithm without LLC, and the theoretical findings under LLC.

### Dependencies
You can simply run the following command automatically install the dependencies

```pip install -r requirement.txt ```

This code mainly requires the following:
- Python 3.*
- tqdm
- Pytorch
- [higher](https://github.com/facebookresearch/higher) 
- [Torchmeta](https://github.com/tristandeleu/pytorch-meta) 

###  Data Preparation

You can download the  [Omniglot](https://github.com/brendenlake/omniglot), 
[MiniImagenet](https://github.com/renmengye/few-shot-ssl-public/), [TieredImagenet](https://github.com/renmengye/few-shot-ssl-public/), 
[Mnist](http://yann.lecun.com/exdb/mnist/) and FashionMnist dataset from the attached link, and put the dataset in the corresponding `/data/*` folder.

### Usage

You can run the python file for different applications following the script below:

```
Python Few_shot.py  # For few shot classification tasks.
Python  Data_hyper_cleaning.py  # For data hyper-cleaning tasks.
Python  Numerical.py  # For the non-convex numerical examples.
```

### Citation

If you use IAPTT-GM for academic research, you are highly encouraged to cite the following paper:
- Risheng Liu, Yaohua Liu, Shangzhi Zeng, Jin Zhang. ["Towards Gradient-based Bilevel Optimization with Non-convex Followers and Beyond"](https://arxiv.org/abs/2110.00455). NeurIPS, 2021.