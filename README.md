# Torwards Gradient-based Bilevel Optimization with non-convex Followers and Beyond
This repo contains code accompaning the paper, Torwards Gradient-based Bilevel Optimization with non-convex Followers and Beyond (Liu et al., NeurIPS 2021). It includes code for running the numerical example, few-shot classification and data hyper-cleaning experiments.

## Abstract
In recent years, Bi-Level Optimization (BLO) techniques have received extensive attentions from both learning and vision communities. A variety of BLO models in complex and practical tasks are of non-convex follower structure in nature (a.k.a., without Lower-Level Convexity, LLC for short). However, this challenging class of BLOs is lack of developments on both efficient solution strategies and solid the oretical guarantees. In this work, we propose a new algorithmic framework, named Initialization Auxiliary and Pessimistic Trajectory Truncated Gradient Method (IAPTT-GM), to partially address the above issues. In particular, by introducing an auxiliary as initialization to guide the optimization dynamics and designing a pessimistic trajectory truncation operation, we construct a reliable approximate version of the original BLO in the absence of LLC hypothesis. Our theoretical investigations establish the convergence of solutions returned by IAPTT-GM towards those of the original BLO without LLC. As an additional bonus, we also theoretically justify the quality of our IAPTT-GM embedded with Nesterov’s accelerated dynamics under LLC. The experimental results confirm both the convergence of our algorithm without LLC, and the theoretical findings under LLC.

## Dependencies
You can simply run the following command automatically install the dependencies
'pip install -r requirement.txt '

This code mainly requires the following:
- Python 3.*
- higher 
- tqdm
- numpy
- Pytorch
- torchmeta


## 
