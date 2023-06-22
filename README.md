# Active Cost-aware Labeling of Streaming Data
This GitHub contains the code to run simulations for the paper "Active Cost-aware Labeling of Streaming Data". In this paper, we investigate how to actively label streaming data when labeling each data is expensive. We consider the problem in two settings: the K discrete type setting and the continuous setting. For the K discrete setting, we propose an algorithm inspired by the UCB algorithm and give a matching lower bound and upper bound. For the continuous setting, we model the underlying mapping between the data and label by Gaussian Process and propose an algorithm with sublinear loss. We substantiate our theoretical results via extensive experiments on both synthetic and real-world data. 

The code is divided into the experiment part (in experiments.ipynb) and the figure plotting part (in plotting.ipynb). For the experiment part, the code for generating the synthetic data and downloading one of our real world datasets (Supernova dataset) is included in experiments.ipynb. The other real-world dataset (Parkinson's dataset[1]) is provided and the code to clean the dataet is included in experiments.ipynb as well.

If you want to use any part of our code, please cite our paper as
```
@inproceedings{cai2023active,
  title={Active Cost-aware Labeling of Streaming Data},
  author={Cai, Ting and Kandasamy, Kirthevasan},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={9117--9136},
  year={2023},
  organization={PMLR}
}
```

# References
[1] Athanasios Tsanas, Max Little, Patrick McSharry, and Lorraine Ramig. Accurate telemonitoring of Parkinson’s disease progression by non-invasive speech tests. Nature Precedings, pages 1–1, 2009.
