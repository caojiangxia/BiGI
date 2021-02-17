BiGI
===

The source code of BiGI: ”Bipartite Graph Embedding via Mutual Information Maximization" which accepted in WSDM 2021 by Jiangxia Cao*, Xixun Lin*, Shu Guo, Luchen Liu, Tingwen Liu, Bin Wang (* means equal contribution).


```
@inproceedings{bigi2021,
  title={Bipartite Graph Embedding via Mutual Information
Maximization},
  author={Cao*, Jiangxia and Lin*, Xixun and Guo, Shu and Liu, Luchen and Liu, Tingwen and Wang, Bin},
  booktitle={ACM International Conference on Web Search and Data Mining (WSDM)},
  year={2021}
}
```


Requirements
---

Python=3.6.2

PyTorch=1.1.0

CUDA=9.0

Scikit-Learn = 0.22

Scipy = 1.3.1

Preparation
---

Some datasets have been included in the `./dataset` directory. Other datasets can be downloaded from the [official website](https://grouplens.org/datasets/movielens/).

Usage
---

To run this project, please make sure that you have the following packages being downloaded. Our experiments are conducted on a PC with an Intel Xeon E5 2.1GHz CPU and a Tesla V100 GPU.

For running DBLP:

```shell
CUDA_VISIBLE_DEVICES=1 nohup python -u train_rec.py --id dblp --struct_rate 0.00001 --GNN 2 > BiGIdblp.log 2>&1&
```

For running ML-100K:

```shell
CUDA_VISIBLE_DEVICES=1 nohup python -u train_rec.py --data_dir dataset/movie/ml-100k/1/ --batch_size 128 --id ml100k --struct_rate 0.0001 --GNN 2 > BiGI100k.log 2>&1&
```

For running ML-10M:

```shell
CUDA_VISIBLE_DEVICES=1 nohup python -u train_rec.py --batch_size 100000 --data_dir dataset/movie/ml-10m/ml-10M100K/1/ --id ml10m --struct_rate 0.00001 > BiGI10m.log 2>&1&
```

For running Wiki(5:5):

```shell
CUDA_VISIBLE_DEVICES=1 nohup python -u train_lp.py --id wiki5 --struct_rate 0.0001 --GNN 2 > BiGIwiki5.log 2>&1&
```

For running Wiki(4:6):

```shell
CUDA_VISIBLE_DEVICES=1 nohup python -u train_lp.py --data_dir dataset/wiki/4/ --id wiki4 --struct_rate 0.0001 --GNN 2 > BiGIwiki4.log 2>&1&
```

