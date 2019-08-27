import numpy as np
import torch
import gym
import pandas as pd
from src.models import QNet
from src.config import mountaincar_config
from src.train_pipeline import train_pipeline
from src.utils import load_qnet, error_info
from src.utils import load_qnet, error_info_step
from collections import deque
from joblib import Parallel, delayed

# if gpu is to be used
# use_cuda = torch.cuda.is_available()
# FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
# ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
# Tensor = FloatTensor

# if gpu is to be used
#use_cuda = torch.cuda.is_available()
use_cuda = False
print(use_cuda)
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor




def parallel_train_pipeline(config, methods, env, eval_qnet, seedvec, max_name_length):
    num_method = len(methods)
    mse = np.zeros(len(methods))
    ind_mse = np.zeros(len(methods))

    results, target = train_pipeline(env, config, eval_qnet, seedvec)

    for i_method in range(num_method):
        mse_1, mse_2 = error_info(results[i_method], target, methods[i_method].ljust(max_name_length))
        mse[i_method] = mse_1
        ind_mse[i_method] = mse_2

    return(mse, ind_mse)


if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    config = mountaincar_config
    eval_qnet = QNet(config.state_dim,config.dqn_hidden_dims,config.action_size)
    load_qnet(eval_qnet,filename='qnet_mc.pth.tar')
    eval_qnet.eval()

    bhv_qnet = QNet(config.state_dim, config.dqn_hidden_dims, config.action_size)
    load_qnet(bhv_qnet, filename='qnet_mc.pth.tar') # target policy
    bhv_qnet.eval() # 読み込んだモデルのモードを切り替える

    methods = ['Model', 'DR', 'DML_RepBM', 'DML_RepBM_estpz', 'DML_RepBM_estpz_wis','DML_RepBM_estpz_sis',  'DML_RepBM_estpz_swis',
               'DML-DR-CROSS-K-ND', 'dml_dr_cross_k_estpz_nd', 'dml_dr_cross_k_estpz_wis_nd', 'dml_dr_cross_k_estpz_sis_nd', 'dml_dr_cross_k_estpz_swis_nd','dml_dr_cross_k_chunk_nd',
               'WDR', 'Soft DR', 'Soft WDR',
               'Model Bsl', 'DR Bsl', 'DR EstPz Bsl', 'WDR EstPz Bsl','WDR Bsl', 'Soft DR Bsl', 'Soft WDR Bsl',
               'Model MSE','DR MSE', 'WDR MSE', 'Soft DR MSE', 'Soft WDR MSE',

               'dr_estpz', 'wdr_estpz', 'sdr_estpz', 'swdr_estpz',
               'dr_msepi_estpz', 'wdr_msepi_estpz', 'sdr_msepi_estpz', 'swdr_msepi_estpz',

               'MRDR Q', 'MRDR', 'WMRDR', 'Soft MRDR', 'Soft WMRDR',
               'MRDR-w Q', 'MRDR-w', 'WMRDR-w', 'Soft MRDR-w', 'Soft WMRDR-w',
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
    np.random.seed(seed=100)
    seedvec = np.random.randint(0, config.MAX_SEED, config.sample_num_traj_eval)

    num_method = len(methods)
    max_name_length = len(max(methods,key=len))
    result_parallel = Parallel(n_jobs=-1)([delayed(parallel_train_pipeline)(config, methods, env, eval_qnet, bhv_qnet, seedvec, max_name_length) for i in range(config.N)])
    mse = np.vstack(x[0] for x in result_parallel)
    mse_ind = np.vstack(x[1] for x in result_parallel)
    mse_w = np.vstack(x[2] for x in result_parallel)

    mse_mean = mse.mean(0)
    mse_ind_mean = mse_ind.mean(0)
    mse_w_mean = mse_w.mean(0)

    mse_sd = mse.std(0)
    mse_ind_sd = mse_ind.std(0)
    mse_w_sd = mse_w.std(0)

    mse_result = []
    mse_table = np.zeros((num_method,4))
    print('Average result over {} runs:'.format(config.N))
    for i in range(num_method):
        print('{}: Root mse of mean is {:.3e}±{:.2e}, root mse of individual is {:.3e}±{:.2e}'
              .format(methods[i].ljust(max_name_length), np.sqrt(mse_mean[i]), np.sqrt(mse_mean[i]),
                      np.sqrt(mse_ind_mean[i]), np.sqrt(mse_ind_sd[i])))
        mse_table[i, 0] = np.sqrt(mse_mean[i])
        mse_table[i, 1] = np.sqrt(mse_sd[i])
        mse_table[i, 2] = np.sqrt(mse_ind_mean[i])
        mse_table[i, 3] = np.sqrt(mse_ind_sd[i])

        out = {"method": methods[i], "rmse_mean":np.sqrt(mse_mean[i]), "rmse_sd":np.sqrt(mse_sd[i]), "rmse_ind_mean":np.sqrt(mse_ind_mean[i]), "rmse_ind_sd":np.sqrt(mse_ind_sd[i]) ,"rmse_w_mean":np.sqrt(mse_w_mean[i]), "rmse_w_sd":np.sqrt(mse_w_sd[i]) }
        mse_result.append(out)
    result_df = pd.DataFrame(mse_result)
    result_df.to_csv("mm_result_df.csv")

    mse = [x[0] for x in result_parallel]
    mse_ind = [x[1] for x in result_parallel]
    mse_w = [x[2] for x in result_parallel]
    mse_df = pd.DataFrame({"mse": mse, "mse_ind": mse_ind, "mse_w": mse_w})
    mse_df.to_csv("mm_mse_df.csv")
    #
    #
    # np.random.seed(seed=100)
    # seedvec = np.random.randint(0, config.MAX_SEED, config.sample_num_traj_eval)
    #
    # num_method = len(methods)
    # max_name_length = len(max(methods,key=len))
    # result_parallel = Parallel(n_jobs=-1)([delayed(parallel_train_pipeline)(config, methods, env, eval_qnet, seedvec, max_name_length) for i in range(config.N)])
    # mse = np.vstack(x[0] for x in result_parallel)
    # mse_ind = np.vstack(x[1] for x in result_parallel)
    #
    # mse_mean = mse.mean(0)
    # mse_ind_mean = mse_ind.mean(0)
    #
    # mse_sd = mse.std(0)
    # mse_ind_sd = mse_ind.std(0)
    #
    # mse_result = []
    # mse_table = np.zeros((num_method,4))
    # print('Average result over {} runs:'.format(config.N))
    # for i in range(num_method):
    #     print('{}: Root mse of mean is {:.3e}±{:.2e}, root mse of individual is {:.3e}±{:.2e}'
    #           .format(methods[i].ljust(max_name_length), np.sqrt(mse_mean[i]), np.sqrt(mse_mean[i]),
    #                   np.sqrt(mse_ind_mean[i]), np.sqrt(mse_ind_sd[i])))
    #     mse_table[i, 0] = np.sqrt(mse_mean[i])
    #     mse_table[i, 1] = np.sqrt(mse_sd[i])
    #     mse_table[i, 2] = np.sqrt(mse_ind_mean[i])
    #     mse_table[i, 3] = np.sqrt(mse_ind_sd[i])
    #
    #     out = {"method": methods[i], "rmse_mean":np.sqrt(mse_mean[i]), "rmse_sd":np.sqrt(mse_sd[i]), "rmse_ind_mean":np.sqrt(mse_ind_mean[i]), "rmse_ind_sd":np.sqrt(mse_ind_sd[i]) }
    #     mse_result.append(out)
    # result_df = pd.DataFrame(mse_result)
    # result_df.to_csv("result_df.csv")
    # np.savetxt('result_cartpole.txt', mse_table, fmt='%.3e', delimiter=',')





    #
    # methods = ['Model', 'DR', 'WDR', 'Soft DR', 'Soft WDR',
    #            'Model Bsl', 'DR Bsl', 'WDR Bsl', 'Soft DR Bsl', 'Soft WDR Bsl',
    #            'Model MSE', 'DR MSE', 'WDR MSE', 'Soft DR MSE', 'Soft WDR MSE',
    #            'MRDR Q', 'MRDR', 'WMRDR', 'Soft MRDR', 'Soft WMRDR',
    #            'MRDR-w Q', 'MRDR-w', 'WMRDR-w', 'Soft MRDR-w', 'Soft WMRDR-w',
    #            'IS', 'WIS', 'Soft IS', 'Soft WIS', 'PDIS', 'WPDIS', 'Soft PDIS', 'Soft WPDIS']
    # num_method = len(methods)
    # max_name_length = len(max(methods,key=len))
    #
    # mse = [deque() for method in methods]
    # ind_mse = [deque() for method in methods]
    #
    # for i_run in range(config.N):
    #     print('Run: {}'.format(i_run+1))
    #     results, target = train_pipeline(env,config,eval_qnet)
    #     for i_method in range(num_method):
    #         #print(results[i_method])
    #         mse_1, mse_2 = error_info(results[i_method], target, methods[i_method].ljust(max_name_length))
    #         mse[i_method].append(mse_1)
    #         ind_mse[i_method].append(mse_2)
    #
    # mse_table = np.zeros((num_method,4))
    # print('Average result over {} runs:'.format(config.N))
    # for i in range(num_method):
    #     print('{}: Root mse of mean is {:.3e}±{:.2e}, root mse of individual is {:.3e}±{:.2e}'
    #           .format(methods[i].ljust(max_name_length), np.sqrt(np.mean(mse[i])), np.sqrt(np.std(mse[i])),
    #                   np.sqrt(np.mean(ind_mse[i])), np.sqrt(np.std(ind_mse[i]))))
    #     mse_table[i, 0] = np.sqrt(np.mean(mse[i]))
    #     mse_table[i, 1] = np.sqrt(np.std(mse[i]))
    #     mse_table[i, 2] = np.sqrt(np.mean(ind_mse[i]))
    #     mse_table[i, 3] = np.sqrt(np.std(ind_mse[i]))
    # np.savetxt('results/result_mountaincar.txt', mse_table, fmt='%.3e', delimiter=',')
