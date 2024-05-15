#!/usr/bin/env bash

data_path='data/PeMS/new.csv' #path to the MTS data
adj_path='data/PeMS/histcorr06_new.csv' #'data/PeMS/W_228.csv'  #path to the adjacency matrix, None if not exists
data_root='data/PeMS' #Directory to the MTS data

stamp_path="${data_root}/time_stamp.npy"
#training model
python3 main_stamp.py train --device=0 --n_route=361 --n_his=5 --n_pred=5 --n_train=34 --n_val=5 --n_test=5 --mode=1 --name='PeMS'\
    --ablation='TPGNN' --data_path=$data_path --adj_matrix_path=$adj_path --stamp_path=$stamp_path