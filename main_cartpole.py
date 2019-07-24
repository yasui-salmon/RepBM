# coding: UTF-8

import numpy as np
import torch
import gym
from src.models import QNet
from src.config import cartpole_config
from src.train_pipeline import train_pipeline
from src.utils import load_qnet, error_info
from src.utils import load_qnet, error_info_step
from collections import deque

# if gpu is to be used
#use_cuda = torch.cuda.is_available()
use_cuda = False
print(use_cuda)
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    config = cartpole_config
    eval_qnet = QNet(config.state_dim, config.dqn_hidden_dims, config.action_size)
    load_qnet(eval_qnet, filename='qnet_cp_short.pth.tar') # target policy
    eval_qnet.eval() # 読み込んだモデルのモードを切り替える

    methods = ['Model', 'DR',
               'DML-DR-CROSS-K-ND', 'dml_dr_cross_k_estpz_nd', 'dml_dr_cross_k_estpz_wis_nd','dml_dr_cross_k_chunk_nd',
               'WDR', 'Soft DR', 'Soft WDR',
               'Model Bsl', 'DR Bsl', 'DR EstPz Bsl', 'WDR EstPz Bsl','WDR Bsl', 'Soft DR Bsl', 'Soft WDR Bsl',
               'Model MSE',
               'IS', 'WIS', 'Soft IS', 'Soft WIS', 'PDIS', 'WPDIS', 'Soft PDIS', 'Soft WPDIS']

    # methods = ['Model', 'DR',
    #            'DML-DR-CROSS-K', 'dml_dr_cross_k_estpz','dml_dr_cross_k_estpz_wis',
    #            'DML-DR-CROSS-K-ND', 'dml_dr_cross_k_estpz_nd', 'dml_dr_cross_k_estpz_wis_nd',
    #            'WDR', 'Soft DR', 'Soft WDR',
    #            'Model Bsl', 'DR Bsl', 'DR Bsl estpz', 'WDR Bsl', 'Soft DR Bsl', 'Soft WDR Bsl',
    #            'Model MSE', 'DR MSE', 'WDR MSE', 'Soft DR MSE', 'Soft WDR MSE',
    #            'MRDR Q', 'MRDR', 'WMRDR', 'Soft MRDR', 'Soft WMRDR',
    #            'MRDR-w Q', 'MRDR-w', 'WMRDR-w', 'Soft MRDR-w', 'Soft WMRDR-w',
    #            'IS', 'WIS', 'Soft IS', 'Soft WIS', 'PDIS', 'WPDIS', 'Soft PDIS', 'Soft WPDIS']



    num_method = len(methods)
    max_name_length = len(max(methods,key=len))

    mse = [deque() for method in methods]
    ind_mse = [deque() for method in methods]

    eval_step = 10
    eval_step_list = np.arange(eval_step, config.eval_num_traj, eval_step)
    eval_step_result = []

    for i_run in range(config.N): # N回評価する
        print('Run: {}'.format(i_run+1))
        results, target = train_pipeline(env, config, eval_qnet)

        for i_method in range(num_method):
            mse_1, mse_2 = error_info(results[i_method], target, methods[i_method].ljust(max_name_length))
            mse[i_method].append(mse_1)
            ind_mse[i_method].append(mse_2)


        for i_method in range(num_method):
            for step in eval_step_list:
                mse_1, mse_2 = error_info_step(results[i_method], target, step)
                eval_out = {"i_run":i_run, "method": methods[i_method].ljust(max_name_length), "step": step, "mse_1": mse_1, "mse_2": mse_2}
                eval_step_result.append(eval_out)

    import pandas as pd
    pd.DataFrame(eval_step_result).to_csv("results/step_eval.csv")

    mse_table = np.zeros((num_method,4))
    print('Average result over {} runs:'.format(config.N))
    for i in range(num_method):
        print('{}: Root mse of mean is {:.3e}±{:.2e}, root mse of individual is {:.3e}±{:.2e}'
              .format(methods[i].ljust(max_name_length), np.sqrt(np.mean(mse[i])), np.sqrt(np.std(mse[i])),
                      np.sqrt(np.mean(ind_mse[i])), np.sqrt(np.std(ind_mse[i]))))
        mse_table[i, 0] = np.sqrt(np.mean(mse[i]))
        mse_table[i, 1] = np.sqrt(np.std(mse[i]))
        mse_table[i, 2] = np.sqrt(np.mean(ind_mse[i]))
        mse_table[i, 3] = np.sqrt(np.std(ind_mse[i]))
    np.savetxt('results/result_cartpole.txt', mse_table, fmt='%.3e', delimiter=',')
