import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import models
import sys
import os
import os.path
import time
from models import predict, predict_stamp
from data import STAGNN_Dataset, STAGNN_stamp_Dataset
from torch.utils.tensorboard import SummaryWriter
from utils.utils import evaluate_metric, weight_matrix, weight_matrix_nl, laplacian, vendermonde
from config import DefaultConfig, Logger
from sklearn.metrics import mean_absolute_percentage_error


opt = DefaultConfig()

sys.stdout = Logger(opt.record_path)

# random seed
seed = opt.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

def load(**kwargs):
    opt.parse(kwargs)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.device)

    if opt.adj_matrix_path != None:
        opt.dis_mat = weight_matrix_nl(opt.adj_matrix_path, epsilon=opt.eps)
        opt.dis_mat = torch.from_numpy(opt.dis_mat).float().cuda()
    else:
        opt.dis_mat = 0.0

    # path
    opt.prefix = 'log/' + opt.name + '/'
    if not os.path.exists(opt.prefix):
        os.makedirs(opt.prefix)
    opt.checkpoint_temp_path = opt.prefix + '/temp.pth'
    opt.checkpoint_best_path = opt.prefix + '/best.pth'
    opt.tensorboard_path = opt.prefix
    opt.record_path = opt.prefix + 'record.txt'

    checkpoint_temp_path = opt.checkpoint_temp_path
    checkpoint_temp_path = 'temp_{}.pth'.format(opt.ablation)

    if opt.ablation == 'TPGNN':
        opt.ablation = 'WOTPG'
    elif opt.ablation == 'WOTPG':
        opt.ablation = 'TPGNN'

    opt.output()

    # load data
    batch_size = opt.batch_size
    train_dataset = STAGNN_stamp_Dataset(opt, train=True, val=False)
    val_dataset = STAGNN_stamp_Dataset(opt, train=False, val=True)
    test_dataset = STAGNN_stamp_Dataset(opt, train=False, val=False)
    train_iter = torch.utils.data.DataLoader(
        train_dataset, batch_size, shuffle=True, drop_last=True)
    val_iter = torch.utils.data.DataLoader(val_dataset, batch_size, drop_last=True)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size, drop_last=True)

    # mask
    n_route = opt.n_route
    n_his = opt.n_his
    n_pred = opt.n_pred
    enc_spa_mask = torch.ones(1, 1, n_route, n_route).cuda()
    enc_tem_mask = torch.ones(1, 1, n_his, n_his).cuda()
    dec_slf_mask = torch.tril(torch.ones(
        (1, 1, n_pred + 1, n_pred + 1)), diagonal=0).cuda()
    dec_mul_mask = torch.ones(1, 1, n_pred + 1, n_his).cuda()

    # loss
    loss_fn = nn.L1Loss()

    MAEs, MAPEs, RMSEs = [], [], []

    model = getattr(models, opt.model)(
        opt,
        enc_spa_mask, enc_tem_mask,
        dec_slf_mask, dec_mul_mask
    )
    model.cuda()

    # optimizer
    lr = opt.lr
    if opt.adam['use']:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=opt.adam['weight_decay'])

    # scheduler
    if opt.slr['use']:
        step_size, gamma = opt.slr['step_size'], opt.slr['gamma']
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma)
    elif opt.mslr['use']:
        milestones, gamma = opt.mslr['milestones'], opt.mslr['gamma']
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones, gamma=gamma)

    start_epoch = opt.start_epoch
    min_val_loss = np.inf


    checkpoint = torch.load(checkpoint_temp_path)
    model.load_state_dict(checkpoint['model'])

    MAE, MAPE, RMSE = evaluate_metric(model, test_iter, opt)
    print("MAE:", MAE, ", MAPE:", MAPE, "%, RMSE:", RMSE)

    pred_arr = []
    true_arr = []

    for x, stamp, y in test_iter:
        x, stamp, y = x.cuda(), stamp.cuda(), y.cuda()
        x = x.type(torch.cuda.FloatTensor)
        stamp = stamp.type(torch.cuda.LongTensor)
        y = y.type(torch.cuda.FloatTensor)

        y_true = y.cpu().numpy().reshape(-1, 361)
        y_true = opt.scaler.inverse_transform(y_true)
        y_true = y_true.reshape(opt.batch_size, opt.n_route, opt.n_pred, 1)

        true_arr.append(y_true[:,0,2,0])

        y_pred = predict_stamp(model, x, stamp, y, opt)

        y_pred = y_pred.cpu().numpy()
        y_pred = y_pred.reshape(-1, 361)
        y_pred = opt.scaler.inverse_transform(y_pred)
        y_pred = y_pred.reshape(opt.batch_size, opt.n_route, opt.n_pred, 1)
        pred_arr.append(y_pred[:,0,2,0])

    pred_arr = np.concatenate(pred_arr, axis = 0)
    true_arr = np.concatenate(true_arr, axis = 0)

    np.save('pred.npy', pred_arr)
    np.save('true.npy', true_arr)

if __name__ == '__main__':
    import fire
    fire.Fire()
