# Active Cost-aware Labeling of Streaming Data
This github contains the code to run simulations for the paper "Active Cost-aware Labeling of Streaming Data". In this paper, we investigate how to actively label streaming data when labeling each data is expensive. We consider the problem in two settings: the K discrete type setting and continuous setting. For the K discrete setting, we propose an algorithm inspired by the UCB algorithm and give matching lower bound and upper bound. For the continous setting, we model the underlying mapping between the data and label by Gaussian Process and propose an algorithm which has sublinear loss. We substantiate our theoretical results via extensive experiments on both synthetic and real world data. 

The code is divided into the experiment part (in experiments.ipynb) and the figure plotting part (in plotting.ipynb). For the experiment part, the code for generating the synthetic data and downloading one of our real world datasets (Supernova dataset) is included in experiments.ipynb. The other real world dataset (Parkinsons dataset[1]) is provided and the code to clean the dataet in included in experiments.ipynb as well.

If you want to use any part of our code, please cite our paper.

# References
[1] Athanasios Tsanas, Max Little, Patrick McSharry, and Lorraine Ramig. Accurate telemonitoring of parkinson’s disease progression by non-invasive speech tests. Nature Precedings, pages 1–1, 2009.
