### Model built using Multivariate Time-Series Forecasting with Temporal Polynomial Graph Neural Networks from Liu et al. (2022)
# Train your model for financial data using polynomial graphs

TPGNN
Multivariate Time-Series Forecasting with Temporal Polynomial Graph Neural Networks (NeurIPS 2022)
loadModel_multi.py: Load the multi-step model and implement predictions on the test set
loadModel_single.py: Load the single-step model and implement predictions on the test set
main_single_stamp.py: Main model for single step prediction
models/SubLayers.py Implemented multiple ablation models and matched different ablation models through dictionaries

train model：

### Usage

sh ./scripts/genstamp.sh
```

Starting Training Process
a
```bash
sh ./scripts/multi.sh
```

load model：

python3 loadModel_multi.py load --device=0 --n_route=361 --n_his=5 --n_pred=5 --n_train=34 --n_val=5 --n_test=5 --mode=1 --name='PeMS' --ablation='TPGNN' --data_path='data/PeMS/new.csv' --adj_matrix_path='data/PeMS/histcorr06_new.csv' --stamp_path='data/PeMS/time_stamp.npy

Reference: 
@inproceedings{
liu2022multivariate,
title={Multivariate Time-Series Forecasting with Temporal Polynomial Graph Neural Networks},
author={Yijing Liu and Qinxian Liu and Jian-Wei Zhang and Haozhe Feng and Zhongwei Wang and Zihan Zhou and Wei Chen},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=pMumil2EJh}
}
