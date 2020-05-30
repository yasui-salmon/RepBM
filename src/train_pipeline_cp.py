import time
from src.models import QtNet, MDPnet, Policynet, TerminalClassifier
from src.memory import MRDRTransition, TrajectorySet, SampleSet
from src.utils import *
from collections import deque
import torch.optim as optim

from src.config import gpu_config

import pdb

# if gpu is to be used
if gpu_config.gpu_false_enforce == True:
    use_cuda = False
else:
    use_cuda = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


# Implementation of more robust doubly robust


def mrdr_loss(q_values, onehot_rewards, weights, pib, pie):
    batch_size = q_values.size()[0]
    action_size = q_values.size()[1]
    Omega = FloatTensor(batch_size, action_size, action_size).zero_()
    for i in range(action_size):
        Omega[:, i, i] = 1 / pib.detach()[:, i]
    Omega = Omega - torch.ones(batch_size, action_size, action_size)

    q = pie * q_values - onehot_rewards
    q = q.unsqueeze(2)  # batch_size x action_size x 1

    error = torch.bmm(Omega, q)
    error = torch.bmm(torch.transpose(q, 1, 2), error)
    error = error.squeeze()
    loss = error * weights
    loss = loss.mean(0)
    return loss


def mrdr_train(samples, qnet, optimizer, loss_mode, config, permutation, i_batch, wis_reward=False):
    qnet.train()
    start = (i_batch * config.mrdr_batch_size) % len(permutation)
    if start + config.mrdr_batch_size >= len(permutation):
        start = 0
    transitions = [samples[i] for i in permutation[start:start + config.mrdr_batch_size]]
    num_samples = config.mrdr_batch_size
    batch = MRDRTransition(*zip(*transitions))

    states = torch.cat(batch.state)
    times = torch.cat([Tensor([[t.time]]) for t in transitions]) / config.max_length

    actions = torch.cat(batch.action)  # batch_size x 1
    pib = torch.cat(batch.pib)  # batch_size x action_size
    if loss_mode == 's':
        pie = torch.cat(batch.soft_pie)
    else:
        pie = torch.cat(batch.pie)
    # R_{0:t}
    if wis_reward:
        acc_rewards = torch.cat(batch.wacc_reward) / config.max_length  # batch_size x,
    else:
        acc_rewards = torch.cat(batch.acc_reward) / config.max_length  # batch_size x,
    onehot_rewards = FloatTensor(num_samples, config.action_size).zero_()  # batch_size x action_size
    onehot_rewards.scatter_(1, actions, acc_rewards.unsqueeze(1))

    # w_{0:t-1}^2 * w_t
    w_0tot = torch.cat([t.acc_soft_isweight for t in transitions])
    w_t = torch.cat([t.soft_isweight for t in transitions])
    weights = w_0tot ** 2 / w_t  # batch_size x,

    # Q(s_t, .)
    q_values = qnet(states.type(Tensor), times)  # batch x action_size
    if loss_mode == 's':  # stochastic: add some noise into pie to create a biased but lower variance target
        loss = mrdr_loss(q_values, onehot_rewards, weights, pib, pie)
    elif loss_mode == 'd':  # deterministic: original, deterministic pie
        pib_a = pib.gather(1, actions).squeeze()
        weights = (weights * (1 - pib_a) / (pib_a ** 2))
        qa_values = q_values.gather(1, actions).squeeze()
        loss = weighted_mse_loss(qa_values, acc_rewards, weights)
    elif loss_mode == 'b':  # behavior: use pib as pie in the MRDR objective function
        qa_values = q_values.gather(1, actions).squeeze()
        loss = torch.nn.MSELoss()(qa_values, acc_rewards)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def mrdr_test(samples, qnet, loss_mode, config, permutation, i_batch, wis_reward=False):
    qnet.train()
    # transitions = random.sample(samples, config.mrdr_test_batch_size)
    start = (i_batch * config.mrdr_test_batch_size) % len(permutation)
    if start + config.mrdr_test_batch_size >= len(permutation):
        start = 0
    transitions = [samples[i] for i in permutation[start:start + config.mrdr_test_batch_size]]
    num_samples = config.mrdr_test_batch_size
    batch = MRDRTransition(*zip(*transitions))

    states = torch.cat(batch.state)
    times = torch.cat([Tensor([[t.time]]) for t in transitions]) / config.max_length
    actions = torch.cat(batch.action)  # batch_size x 1
    pib = torch.cat(batch.pib)  # batch_size x action_size
    if loss_mode == 's':
        pie = torch.cat(batch.soft_pie)
    else:
        pie = torch.cat(batch.pie)
    # R_{0:t}
    if wis_reward:
        acc_rewards = torch.cat(batch.wacc_reward) / config.max_length  # batch_size x,
    else:
        acc_rewards = torch.cat(batch.acc_reward) / config.max_length  # batch_size x,
    onehot_rewards = FloatTensor(num_samples, config.action_size).zero_()  # batch_size x action_size
    onehot_rewards.scatter_(1, actions, acc_rewards.unsqueeze(1))

    # w_{0:t-1}^2 * w_t
    w_0tot = torch.cat([t.acc_soft_isweight for t in transitions])
    w_t = torch.cat([t.soft_isweight for t in transitions])
    weights = w_0tot ** 2 / w_t  # batch_size x,

    # Q(s_t, .)
    q_values = qnet(states.type(Tensor), times)  # batch x action_size
    if loss_mode == 's':
        loss = mrdr_loss(q_values, onehot_rewards, weights, pib, pie)
    elif loss_mode == 'd':
        pib_a = pib.gather(1, actions).squeeze()
        weights = weights * (1 - pib_a) / (pib_a ** 2)
        qa_values = q_values.gather(1, actions).squeeze()
        loss = weighted_mse_loss(qa_values, acc_rewards, weights)
    elif loss_mode == 'b':
        qa_values = q_values.gather(1, actions).squeeze()
        loss = torch.nn.MSELoss()(qa_values, acc_rewards)
    return loss.item()


# Implementation of RepBM and baseline MDP model


def mdpmodel_train(memory, mdpnet, optimizer, loss_mode, config):

    noise_dim = config.noise_dim
    state_dim = config.state_dim + noise_dim

    mdpnet.train()
    # Adapted from pytorch tutorial example code
    if config.train_batch_size > len(memory):
        return

    transitions = memory.sample(config.train_batch_size)
    time = transitions[0].time
    ratio = 1 / memory.u[time] if memory.u[time] != 0 else 0

    next_states = torch.cat([t.next_state for t in transitions])
    states = torch.cat([t.state for t in transitions])  # batch_size x state_dim
    states_diff = (next_states - states)  # batch_size x state_dim
    actions = torch.cat([t.action for t in transitions])  # batch_size x 1
    reward = torch.cat([t.reward for t in transitions])  # batch_size or batch_size x 1
    reward = reward.squeeze()  # batch_size

    factual = torch.cat([t.factual for t in transitions])
    weights = FloatTensor(config.train_batch_size)
    weights.fill_(1.0)
    weights = weights + factual * ratio
    weights_2 = factual * ratio

    # Compute T(s_t, a) - the model computes T(s_t), then we select the
    # columns of actions taken
    expanded_actions = actions.unsqueeze(2)  # batch_size x 1 x 1
    expanded_actions = expanded_actions.expand(-1, -1, state_dim)  # batch_size x 1 x state_dim

    predict_state_diff, predict_reward_value, rep = mdpnet(states.type(Tensor))
    # predict_state_diff = batch_size x 2 x state_dim
    predict_state_diff = predict_state_diff.gather(1, expanded_actions).squeeze()  # batch_size x state_dim
    predict_reward_value = predict_reward_value.gather(1, actions).squeeze()

    sum_factual = sum([t.factual[0] for t in transitions])
    sum_control = sum([t.last_factual[0] for t in transitions])
    if sum_factual > 0 and sum_control > 0:
        factual_index = LongTensor([i for i, t in enumerate(transitions) if t.factual[0] == 1])
        last_factual_index = LongTensor([i for i, t in enumerate(transitions) if t.last_factual[0] == 1])
        factual_rep = rep.index_select(0, factual_index)
        control_rep = rep.index_select(0, last_factual_index)
        loss_rep = mmd_lin(factual_rep, control_rep)
    else:
        loss_rep = 0

    # Compute loss
    if loss_mode == 0:  # MSE_mu
        loss = torch.nn.MSELoss()(predict_state_diff, states_diff) \
               + torch.nn.MSELoss()(predict_reward_value, reward)

    elif loss_mode == 1:  # MSE_pi + MSE_mu (objective in RepBM paper)
        loss = weighted_mse_loss(predict_state_diff, states_diff, weights) \
               + weighted_mse_loss(predict_reward_value, reward, weights) \
               + config.alpha_rep * loss_rep  # + config.alpha_bce*torch.nn.BCELoss(weights)(predict_soft_done,done)

    elif loss_mode == 2:  # MSE_pi: only MSE_pi, as an ablation study
        loss = weighted_mse_loss(predict_state_diff, states_diff, weights_2) \
               + weighted_mse_loss(predict_reward_value, reward, weights_2)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in mdpnet.parameters():
        if param.grad is not None:
            param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss.item()


def mdpmodel_test(memory, mdpnet, loss_mode, config):
    mdpnet.eval()
    noise_dim = config.noise_dim
    state_dim = config.state_dim + noise_dim

    # Adapted from pytorch tutorial example code
    if config.test_batch_size > len(memory):
        return
    transitions = memory.sample(config.test_batch_size)
    time = transitions[0].time
    ratio = 1 / memory.u[time] if memory.u[time] != 0 else 0

    # Compute a mask of non-final states and concatenate the batch elements
    next_states = torch.cat([t.next_state for t in transitions])
    states = torch.cat([t.state for t in transitions])
    states_diff = (next_states - states)
    actions = (torch.cat([t.action for t in transitions]))
    reward = torch.cat([t.reward for t in transitions])
    reward = reward.squeeze()
    actions_label = actions.squeeze().float()

    sum_factual = sum([t.factual[0] for t in transitions])
    sum_control = sum([t.last_factual[0] for t in transitions])

    factual = torch.cat([t.factual for t in transitions])
    weights = FloatTensor(config.test_batch_size)
    weights.fill_(1.0)
    weights = weights + factual * ratio
    weights_2 = factual * ratio

    # Compute T(s_t, a) - the model computes T(s_t), then we select the
    # columns of actions taken
    expanded_actions = actions.unsqueeze(2)
    expanded_actions = expanded_actions.expand(-1, -1, state_dim)
    predict_state_diff, predict_reward_value, rep = mdpnet(states.type(Tensor))
    predict_state_diff = predict_state_diff.gather(1, expanded_actions).squeeze()
    predict_reward_value = predict_reward_value.gather(1, actions).squeeze()

    if sum_factual > 0 and sum_control > 0:
        factual_index = LongTensor([i for i, t in enumerate(transitions) if t.factual[0] == 1])
        last_factual_index = LongTensor([i for i, t in enumerate(transitions) if t.last_factual[0] == 1])
        factual_rep = rep.index_select(0, factual_index)
        control_rep = rep.index_select(0, last_factual_index)
        loss_rep = mmd_lin(factual_rep, control_rep)
    else:
        loss_rep = 0

    # Compute loss
    loss_mode = 0
    if loss_mode == 0:
        loss = torch.nn.MSELoss()(predict_state_diff, states_diff) + torch.nn.MSELoss()(predict_reward_value, reward)
    elif loss_mode == 1:
        loss = weighted_mse_loss(predict_state_diff, states_diff, weights) + weighted_mse_loss(predict_reward_value,
                                                                                               reward,
                                                                                               weights) + config.alpha_rep * loss_rep
    elif loss_mode == 2:
        loss = weighted_mse_loss(predict_state_diff, states_diff, weights_2) + weighted_mse_loss(predict_reward_value,
                                                                                                 reward, weights_2)

    return loss.item()


def pzmodel_train(memory, mdpnet, optimizer, loss_mode, config):
    mdpnet.train()
    noise_dim = config.noise_dim
    state_dim = config.state_dim + noise_dim

    # Adapted from pytorch tutorial example code
    if config.policy_train_batch_size > len(memory):
        return
    transitions = memory.sample(config.train_batch_size)
    time = transitions[0].time
    ratio = 1 / memory.u[time] if memory.u[time] != 0 else 0

    states = torch.cat([t.state for t in transitions])  # batch_size x state_dim
    actions = torch.cat([t.action for t in transitions])  # batch_size x 1
    actions_label = actions.squeeze().long()  # float変換しない場合 MSELossでは詰まる。が、そもそもMSEで良いのか？

    factual = torch.cat([t.factual for t in transitions])
    weights = FloatTensor(config.train_batch_size)
    weights.fill_(1.0)
    weights = weights + factual * ratio
    weights_2 = factual * ratio

    # Compute T(s_t, a) - the model computes T(s_t), then we select the
    # columns of actions taken
    expanded_actions = actions.unsqueeze(2)  # batch_size x 1 x 1
    expanded_actions = expanded_actions.expand(-1, -1, state_dim)  # batch_size x 1 x state_dim

    # predict_state_diff, predict_reward_value, rep = mdpnet(states.type(Tensor))
    rep, predict_pizero_value = mdpnet(states.type(Tensor))  # output pizero for DML
    predict_pizero_value = predict_pizero_value  # .gather(1, actions).squeeze() #predict pizero for DML

    # sum_factual = sum([t.factual[0] for t in transitions])
    # sum_control = sum([t.last_factual[0] for t in transitions])
    # if sum_factual > 0 and sum_control > 0:
    #     factual_index = LongTensor([i for i, t in enumerate(transitions) if t.factual[0] == 1])
    #     last_factual_index = LongTensor([i for i, t in enumerate(transitions) if t.last_factual[0] == 1])
    #     factual_rep = rep.index_select(0,factual_index)
    #     control_rep = rep.index_select(0,last_factual_index)
    #     loss_rep = mmd_lin(factual_rep, control_rep)
    # else:
    #     loss_rep = 0

    # Compute loss
    # print(predict_pizero_value, actions_label)

    m = torch.nn.LogSoftmax(dim=1)
    loss = torch.nn.NLLLoss()(m(predict_pizero_value), actions_label)  # added pizero prediction

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    for param in mdpnet.parameters():
        if param.grad is not None:
            param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss.item()


def pzmodel_test(memory, mdpnet, loss_mode, config):
    mdpnet.eval()
    noise_dim = config.noise_dim
    state_dim = config.state_dim + noise_dim

    # Adapted from pytorch tutorial example code
    if config.test_batch_size > len(memory):
        return
    transitions = memory.sample(config.test_batch_size)
    time = transitions[0].time
    ratio = 1 / memory.u[time] if memory.u[time] != 0 else 0

    # Compute a mask of non-final states and concatenate the batch elements
    next_states = torch.cat([t.next_state for t in transitions])
    states = torch.cat([t.state for t in transitions])
    states_diff = (next_states - states)
    actions = (torch.cat([t.action for t in transitions]))
    reward = torch.cat([t.reward for t in transitions])
    reward = reward.squeeze()
    actions_label = actions.squeeze().long()

    sum_factual = sum([t.factual[0] for t in transitions])
    sum_control = sum([t.last_factual[0] for t in transitions])

    factual = torch.cat([t.factual for t in transitions])
    weights = FloatTensor(config.test_batch_size)
    weights.fill_(1.0)
    weights = weights + factual * ratio
    weights_2 = factual * ratio

    # Compute T(s_t, a) - the model computes T(s_t), then we select the
    # columns of actions taken
    expanded_actions = actions.unsqueeze(2)
    expanded_actions = expanded_actions.expand(-1, -1, state_dim)
    rep, predict_pizero_value = mdpnet(states.type(Tensor))
    # predict_pizero_value = predict_pizero_value#.gather(1, actions).squeeze()  # predict pizero for DML

    if sum_factual > 0 and sum_control > 0:
        factual_index = LongTensor([i for i, t in enumerate(transitions) if t.factual[0] == 1])
        last_factual_index = LongTensor([i for i, t in enumerate(transitions) if t.last_factual[0] == 1])
        factual_rep = rep.index_select(0, factual_index)
        control_rep = rep.index_select(0, last_factual_index)
        loss_rep = mmd_lin(factual_rep, control_rep)
    else:
        loss_rep = 0

    # Compute loss
    loss = torch.nn.CrossEntropyLoss()(predict_pizero_value, actions_label)

    return loss.item()


def terminal_classifier_train(memory, model, optimizer, batch_size):
    model.train()
    transitions_origin = memory.sample(batch_size)
    transitions_terminal = memory.sample_terminal(batch_size)
    transitions = transitions_origin + transitions_terminal
    num_samples = len(transitions)

    states = torch.cat([t.state for t in transitions])
    next_states = torch.cat([t.next_state for t in transitions])
    all_states = torch.cat((next_states, states), 0)
    done = torch.cat([FloatTensor([[t.done and t.time < 199]]) for t in transitions])
    all_done = torch.cat((done, torch.zeros(num_samples, 1)), 0)
    predict_soft_done = model(all_states.type(Tensor))

    # Optimize the model
    loss = torch.nn.BCEWithLogitsLoss()(predict_soft_done, all_done)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    acc = ((predict_soft_done.detach()[:, 0] > 0).float() == all_done[:, 0]).sum().item() / (all_done.size()[0])
    return loss.item(), acc


def terminal_classifier_test(memory, model, batch_size):
    model.eval()
    transitions_origin = memory.sample(batch_size)
    transitions_terminal = memory.sample_terminal(batch_size)
    transitions = transitions_origin + transitions_terminal
    num_samples = len(transitions)

    states = torch.cat([t.state for t in transitions])
    next_states = torch.cat([t.next_state for t in transitions])
    all_states = torch.cat((next_states, states), 0)
    done = torch.cat([FloatTensor([[t.done and t.time < 199]]) for t in transitions])
    all_done = torch.cat((done, torch.zeros(num_samples, 1)), 0)
    predict_soft_done = model(all_states.type(Tensor))

    # Optimize the model
    loss = torch.nn.BCEWithLogitsLoss()(predict_soft_done, all_done)
    acc = ((predict_soft_done.detach()[:, 0] > 0).float() == all_done[:, 0]).sum().item() / (all_done.size()[0])
    return loss.item(), acc


def rollout_batch(init_states, mdpnet, is_done, num_rollout, policy_qnet, epsilon, action_size, maxlength, config,
                  init_done=None, init_actions=None):

    noise_dim = config.noise_dim
    state_dim = config.state_dim + noise_dim

    ori_batch_size = init_states.size()[0]
    state_dim = init_states.size()[1]
    batch_size = init_states.size()[0] * num_rollout
    init_states = init_states.repeat(num_rollout, 1)
    if init_actions is not None:
        init_actions = init_actions.repeat(num_rollout, 1)
    if init_done is not None:
        init_done = init_done.repeat(num_rollout)
    states = init_states
    if init_done is None:
        done = ByteTensor(batch_size)
        done.fill_(0)
    else:
        done = init_done
    if init_actions is None:
        actions = epsilon_greedy_action_batch(states, policy_qnet, epsilon, action_size)
    else:
        actions = init_actions

    n_steps = 0
    t_reward = torch.zeros(batch_size)
    while not (sum(done.long() == batch_size) or n_steps >= maxlength):  # for Q?
        if n_steps > 0:
            actions = epsilon_greedy_action_batch(states, policy_qnet, epsilon, action_size)

        states_re = Tensor(torch.tensor(np.float32(config.rescale)) * torch.tensor(states))  # rescale state
        states_diff, reward, _ = mdpnet.forward(states_re.type(Tensor))  # result of prediction?
        states_diff = states_diff.detach()
        reward = reward.detach().gather(1, actions).squeeze()
        expanded_actions = actions.unsqueeze(2)
        expanded_actions = expanded_actions.expand(-1, -1, state_dim)

        states_diff = states_diff.gather(1, expanded_actions).squeeze()
        states_diff = states_diff.view(-1, state_dim)
        states_diff = 1 / torch.tensor(np.float32(config.rescale)) * states_diff
        next_states = states_diff + torch.tensor(states)
        states = next_states

        t_reward = t_reward + (1 - done).float() * reward  # total rewardに報酬を加算。 trajectoryが終わったときにはrewardには加算されない。
        done = done | (is_done.forward(states.type(Tensor)).detach()[:, 0] > 0)
        n_steps += 1

    value = t_reward.numpy()
    value = np.reshape(value, [num_rollout, ori_batch_size])
    return np.mean(value, 0)


def reward_prediction(traj_set, mdpnet, max_length, config, soften=False):
    num_samples = len(traj_set)
    traj_len = np.zeros(num_samples, 'int')
    noise_dim = config.noise_dim
    state_dim = config.state_dim + noise_dim

    state_tensor = FloatTensor(num_samples, max_length, state_dim).zero_()
    action_tensor = LongTensor(num_samples, max_length, 1).zero_()
    pie_tensor = FloatTensor(num_samples, max_length, config.action_size).zero_()
    done_tensor = ByteTensor(num_samples, max_length).fill_(1)
    rewards = np.zeros((num_samples, max_length))

    for i_traj in range(num_samples):
        traj_len[i_traj] = len(traj_set.trajectories[i_traj])

        stv = torch.cat([t.state for t in traj_set.trajectories[i_traj]])

        state_tensor[i_traj, 0:traj_len[i_traj], :] = stv  # trajectoryからstateだけ取り出す
        action_tensor[i_traj, 0:traj_len[i_traj], :] = torch.cat(
            [t.action for t in traj_set.trajectories[i_traj]])  # trajectoryからactionだけ取り出す
        done_tensor[i_traj, 0:traj_len[i_traj]].fill_(0)

        if traj_len[i_traj] < config.max_length:
            done_tensor[i_traj, traj_len[i_traj]:].fill_(1)  # まだ終わってないところは１にしておく

        if soften:
            pie_tensor[i_traj, 0:traj_len[i_traj], :] = torch.cat([t.soft_pie for t in traj_set.trajectories[i_traj]])

    # Cut off unnecessary computation: if a time step t
    nonzero_is = np.zeros(config.max_length, 'int')
    for i_traj in range(num_samples):
        w = 1
        for t in traj_set.trajectories[i_traj]:
            if w > 0:
                nonzero_is[t.time] += 1
            w *= t.isweight[0]

    # reward prediction をすべてのtrajectoryのi_step目で算出する
    for i_step in range(max_length):
        actions = action_tensor[:, i_step, :]
        states = state_tensor[:, i_step, :]
        states_re = Tensor(torch.tensor(np.float32(config.rescale)) * torch.tensor(states))  # rescale state

        states_diff, reward, _ = mdpnet.forward(states_re.type(Tensor))
        predict_reward_value = reward.gather(1, actions).squeeze()

        rewards[:, i_step] = predict_reward_value.detach().numpy()

    return (rewards)


def compute_values(traj_set, model, is_done, policy_qnet, config, max_length, model_type='MDP',
                   soften=False):  # ここでpizeroの予測値を返す様にする
    noise_dim = config.noise_dim
    state_dim = config.state_dim + noise_dim

    num_samples = len(traj_set)
    traj_len = np.zeros(num_samples, 'int')
    state_tensor = FloatTensor(num_samples, max_length, state_dim).zero_()
    action_tensor = LongTensor(num_samples, max_length, 1).zero_()
    pie_tensor = FloatTensor(num_samples, max_length, config.action_size).zero_()
    done_tensor = ByteTensor(num_samples, max_length).fill_(1)
    V_value = np.zeros((num_samples, max_length))
    Q_value = np.zeros((num_samples, max_length))

    for i_traj in range(num_samples):
        traj_len[i_traj] = len(traj_set.trajectories[i_traj])

        stv = torch.cat([t.state for t in traj_set.trajectories[i_traj]])

        state_tensor[i_traj, 0:traj_len[i_traj], :] = stv  # trajectoryからstateだけ取り出す
        action_tensor[i_traj, 0:traj_len[i_traj], :] = torch.cat(
            [t.action for t in traj_set.trajectories[i_traj]])  # trajectoryからactionだけ取り出す
        done_tensor[i_traj, 0:traj_len[i_traj]].fill_(0)

        if traj_len[i_traj] < config.max_length:
            done_tensor[i_traj, traj_len[i_traj]:].fill_(1)  # まだ終わってないところは１にしておく

        if soften:
            pie_tensor[i_traj, 0:traj_len[i_traj], :] = torch.cat([t.soft_pie for t in traj_set.trajectories[i_traj]])

    # Cut off unnecessary computation: if a time step t
    nonzero_is = np.zeros(max_length, 'int')
    for i_traj in range(num_samples):
        w = 1
        for t in traj_set.trajectories[i_traj]:
            if w > 0:
                nonzero_is[t.time] += 1
            w *= t.isweight[0]

    # V,Qをすべてのtrajectoryのi_step目で算出する
    for i_step in range(max_length):
        # if nonzero_is[i_step] == 0:
        #     break
        if model_type == 'MDP':  # doubly robustとかが使う
            if soften:
                V_value[:, i_step] = rollout_batch(state_tensor[:, i_step, :], model, is_done, config.eval_num_rollout,
                                                   policy_qnet,
                                                   epsilon=config.soften_epsilon, action_size=config.action_size,
                                                   maxlength=max_length - i_step, config=config,
                                                   init_done=done_tensor[:, i_step])

                Q_value[:, i_step] = rollout_batch(state_tensor[:, i_step, :], model, is_done, config.eval_num_rollout,
                                                   policy_qnet,
                                                   epsilon=config.soften_epsilon, action_size=config.action_size,
                                                   maxlength=max_length - i_step, config=config,
                                                   init_done=done_tensor[:, i_step],
                                                   init_actions=action_tensor[:, i_step, :])
            else:
                V_value[:, i_step] = rollout_batch(state_tensor[:, i_step, :], model, is_done, config.eval_num_rollout,
                                                   policy_qnet, epsilon=0, action_size=config.action_size,
                                                   maxlength=max_length - i_step, config=config,
                                                   init_done=done_tensor[:, i_step])

                Q_value[:, i_step] = rollout_batch(state_tensor[:, i_step, :], model, is_done, config.eval_num_rollout,
                                                   policy_qnet, epsilon=0, action_size=config.action_size,
                                                   maxlength=max_length - i_step, config=config,
                                                   init_done=done_tensor[:, i_step],
                                                   init_actions=action_tensor[:, i_step, :])  # Qの場合にはinit_actionsが付く

        elif model_type == 'Q':
            times = Tensor(num_samples, 1).fill_(i_step) / max_length

            # i_step目のq_valueをmdp_netを使ってすべてのtrajectoryで予測
            q_values = model.forward(state_tensor[:, i_step, :], times).detach() * max_length

            # i_step目のcf policyの行動(greedyなので1 or 0)
            pie_actions = epsilon_greedy_action_batch(state_tensor[:, i_step, :], policy_qnet, 0,
                                                      config.action_size)  # evaluation policy actio(pure greedy)

            if soften:
                zeros_actions = LongTensor(num_samples, 1).zero_()
                ones_actions = 1 - zeros_actions
                V_value[:, i_step] = pie_tensor[:, i_step, 0] * q_values.gather(1, zeros_actions).squeeze() \
                                     + pie_tensor[:, i_step, 1] * q_values.gather(1, ones_actions).squeeze()
            else:
                # Immediate Reward
                # pie_actionsで指定されたindexをq_valueから取り出す
                V_value[:, i_step] = q_values.gather(1, pie_actions).squeeze()

            # Q-value
            # action_tensor(実際にログデータ上に残っているaction)で指定されたindexをq_valueから取り出す
            Q_value[:, i_step] = q_values.gather(1, action_tensor[:, i_step, :]).squeeze()

        elif model_type == 'IS':
            pass
    return V_value, Q_value


def compute_values_dml(traj_set, model, is_done, policy_qnet, config, max_length, model_type='MDP',
                       soften=False):  # ここでpizeroの予測値を返す様にする
    noise_dim = config.noise_dim
    state_dim = config.state_dim + noise_dim

    num_samples = len(traj_set)

    traj_len = np.zeros(num_samples, 'int')
    state_tensor = FloatTensor(num_samples, max_length, state_dim).zero_()
    action_tensor = LongTensor(num_samples, max_length, 1).zero_()
    pie_tensor = FloatTensor(num_samples, max_length, config.action_size).zero_()
    done_tensor = ByteTensor(num_samples, max_length).fill_(1)
    V_value = np.zeros((num_samples, max_length))
    Q_value = np.zeros((num_samples, max_length))

    for i_traj in range(num_samples):
        traj_len[i_traj] = len(traj_set.trajectories[i_traj])

        stv = torch.cat([t.state for t in traj_set.trajectories[i_traj]])

        state_tensor[i_traj, 0:traj_len[i_traj], :] = stv  # trajectoryからstateだけ取り出す
        action_tensor[i_traj, 0:traj_len[i_traj], :] = torch.cat(
            [t.action for t in traj_set.trajectories[i_traj]])  # trajectoryからactionだけ取り出す
        done_tensor[i_traj, 0:traj_len[i_traj]].fill_(0)

        if traj_len[i_traj] < config.max_length:
            done_tensor[i_traj, traj_len[i_traj]:].fill_(1)  # まだ終わってないところは１にしておく

        if soften:
            pie_tensor[i_traj, 0:traj_len[i_traj], :] = torch.cat([t.soft_pie for t in traj_set.trajectories[i_traj]])

    # Cut off unnecessary computation: if a time step t
    nonzero_is = np.zeros(max_length, 'int')
    for i_traj in range(num_samples):
        w = 1
        for t in traj_set.trajectories[i_traj]:
            if w > 0:
                nonzero_is[t.time] += 1
            w *= t.isweight[0]

    # V,Qをすべてのtrajectoryのi_step目で算出する
    for i_step in range(max_length):
        # if nonzero_is[i_step] == 0:
        #     break
        if model_type == 'MDP':  # doubly robustとかが使う
            if soften:
                V_value[:, i_step] = rollout_batch(state_tensor[:, i_step, :], model, is_done, config.eval_num_rollout,
                                                   policy_qnet,
                                                   epsilon=config.soften_epsilon, action_size=config.action_size,
                                                   maxlength=max_length - i_step, config=config,
                                                   init_done=done_tensor[:, i_step])

                Q_value[:, i_step] = rollout_batch(state_tensor[:, i_step, :], model, is_done, config.eval_num_rollout,
                                                   policy_qnet,
                                                   epsilon=config.soften_epsilon, action_size=config.action_size,
                                                   maxlength=max_length - i_step, config=config,
                                                   init_done=done_tensor[:, i_step],
                                                   init_actions=action_tensor[:, i_step, :])
            else:
                V_value[:, i_step] = rollout_batch(state_tensor[:, i_step, :], model, is_done, config.eval_num_rollout,
                                                   policy_qnet, epsilon=0, action_size=config.action_size,
                                                   maxlength=max_length - i_step, config=config,
                                                   init_done=done_tensor[:, i_step])

                Q_value[:, i_step] = rollout_batch(state_tensor[:, i_step, :], model, is_done, config.eval_num_rollout,
                                                   policy_qnet, epsilon=0, action_size=config.action_size,
                                                   maxlength=max_length - i_step, config=config,
                                                   init_done=done_tensor[:, i_step],
                                                   init_actions=action_tensor[:, i_step, :])  # Qの場合にはinit_actionsが付く

        elif model_type == 'Q':
            times = Tensor(num_samples, 1).fill_(i_step) / max_length

            # i_step目のq_valueをmdp_netを使ってすべてのtrajectoryで予測
            q_values = model.forward(state_tensor[:, i_step, :], times).detach() * max_length

            # i_step目のcf policyの行動(greedyなので1 or 0)
            pie_actions = epsilon_greedy_action_batch(state_tensor[:, i_step, :], policy_qnet, 0,
                                                      config.action_size)  # evaluation policy action

            if soften:
                zeros_actions = LongTensor(num_samples, 1).zero_()
                ones_actions = 1 - zeros_actions
                V_value[:, i_step] = pie_tensor[:, i_step, 0] * q_values.gather(1, zeros_actions).squeeze() \
                                     + pie_tensor[:, i_step, 1] * q_values.gather(1, ones_actions).squeeze()
            else:
                # Immediate Reward
                # pie_actionsで指定されたindexをq_valueから取り出す
                V_value[:, i_step] = q_values.gather(1, pie_actions).squeeze()

            # Q-value
            # action_tensor(実際にログデータ上に残っているaction)で指定されたindexをq_valueから取り出す
            Q_value[:, i_step] = q_values.gather(1, action_tensor[:, i_step, :]).squeeze()

        elif model_type == 'IS':
            pass
    return V_value, Q_value


def rollout_batch_pz(init_states, mdpnet, is_done, num_rollout, policy_qnet, epsilon, action_size, maxlength, config,
                     init_done=None, init_actions=None):
    ori_batch_size = init_states.size()[0]
    state_dim = init_states.size()[1]
    batch_size = init_states.size()[0] * num_rollout
    init_states = init_states.repeat(num_rollout, 1)

    if init_actions is not None:
        init_actions = init_actions.repeat(num_rollout, 1)

    if init_done is not None:
        init_done = init_done.repeat(num_rollout)

    states = init_states

    if init_done is None:
        done = ByteTensor(batch_size)
        done.fill_(0)
    else:
        done = init_done

    if init_actions is None:
        actions = epsilon_greedy_action_batch(states, policy_qnet, epsilon, action_size)
    else:
        actions = init_actions

    states_re = Tensor(torch.tensor(np.float32(config.rescale)) * torch.tensor(states))  # rescale state
    rep, pizero = mdpnet.forward(states_re.type(Tensor))  # result of prediction?

    m = torch.nn.Softmax(dim=1)

    pizero_prob_mat = m(pizero.detach())

    action_sq = actions.squeeze()
    idx = np.arange(len(actions.squeeze()))

    # print(pizero_prob_mat) #loop 23くらいから予測値が全部同一になってくる
    pizero_prob_vec = pizero_prob_mat[[idx], [np.array(action_sq)]]

    return pizero_prob_vec


def compute_pizero(traj_set, model, is_done, policy_qnet, config, model_type='MDP', soften=False):
    noise_dim = config.noise_dim
    state_dim = config.state_dim + noise_dim

    num_samples = len(traj_set)
    traj_len = np.zeros(num_samples, 'int')
    state_tensor = FloatTensor(num_samples, config.max_length, state_dim).zero_()
    action_tensor = LongTensor(num_samples, config.max_length, 1).zero_()
    pie_tensor = FloatTensor(num_samples, config.max_length, config.action_size).zero_()
    done_tensor = ByteTensor(num_samples, config.max_length).fill_(1)
    pizero_value = np.zeros((num_samples, config.max_length))

    for i_traj in range(num_samples):
        traj_len[i_traj] = len(traj_set.trajectories[i_traj])
        state_tensor[i_traj, 0:traj_len[i_traj], :] = torch.cat(
            [t.state for t in traj_set.trajectories[i_traj]])  # trajectoryからstateだけ取り出す
        action_tensor[i_traj, 0:traj_len[i_traj], :] = torch.cat(
            [t.action for t in traj_set.trajectories[i_traj]])  # trajectoryからactionだけ取り出す
        done_tensor[i_traj, 0:traj_len[i_traj]].fill_(0)

        if traj_len[i_traj] < config.max_length:
            done_tensor[i_traj, traj_len[i_traj]:].fill_(1)  # まだ終わってないところは１にしておく

        if soften:
            pie_tensor[i_traj, 0:traj_len[i_traj], :] = torch.cat([t.soft_pie for t in traj_set.trajectories[i_traj]])

    # Cut off unnecessary computation: if a time step t
    nonzero_is = np.zeros(config.max_length, 'int')
    for i_traj in range(num_samples):
        w = 1
        for t in traj_set.trajectories[i_traj]:
            if w > 0:
                nonzero_is[t.time] += 1
            w *= t.isweight[0]

    for i_step in range(config.max_length):
        pizero_value[:, i_step] = rollout_batch_pz(state_tensor[:, i_step, :], model, is_done, config.eval_num_rollout,
                                                   policy_qnet, epsilon=config.soften_epsilon,
                                                   action_size=config.action_size,
                                                   maxlength=config.max_length - i_step, config=config,
                                                   init_done=done_tensor[:, i_step],
                                                   init_actions=action_tensor[:, i_step, :])

    return (pizero_value)


def doubly_robust(traj_set, V_value, Q_value, config, wis=False, soften=False):
    num_samples = len(traj_set)
    weights = np.zeros((num_samples, config.max_length))
    weights_sum = np.zeros(config.max_length)

    for i_traj in range(num_samples):
        for n in range(config.max_length):
            if n >= len(traj_set.trajectories[i_traj]):  # trajectoryの長さを超えたらbreak
                weights[i_traj:, n] = weights[i_traj, n - 1]
                break
            if soften:
                weights[i_traj, n] = traj_set.trajectories[i_traj][n].acc_soft_isweight[0].item()
            else:
                # あるtrajectory i_traj における step n においてのimportance samplingをweightに保存する
                weights[i_traj, n] = traj_set.trajectories[i_traj][n].acc_isweight[0].item()

    # WISの場合にはtrajectoryで得られた重みを合計した値で重みを割る
    if wis:
        for n in range(config.max_length):
            weights_sum[n] = np.sum(weights[:, n])
            if weights_sum[n] != 0:
                weights[:, n] = (weights[:, n] * num_samples) / weights_sum[n]

    value = np.zeros(num_samples)
    for i_traj in range(num_samples):
        w = 1
        for t in traj_set.trajectories[i_traj]:
            # trajectory i_traj のstep tでVを計算する
            # t.reward[0].item() : ログ上の報酬
            # Q_value[i_traj,t.time] : Qの予測値
            # V_value[i_traj,t.time] : Vの予測値
            value[i_traj] += weights[i_traj, t.time] * (t.reward[0].item() - Q_value[i_traj, t.time]) + w * V_value[
                i_traj, t.time]
            w = weights[i_traj, t.time]
            if w == 0:
                break

    return value


def doubly_robust_dml_chunk(traj_set, V_value, Q_value, mu_hat, config, wis=False, soften=False):
    num_samples = len(traj_set)
    weights = np.zeros((num_samples, config.max_length))
    weights_sum = np.zeros(config.max_length)

    for i_traj in range(num_samples):
        for n in range(config.max_length):
            if n >= len(traj_set.trajectories[i_traj]):  # trajectoryの長さを超えたらbreak
                weights[i_traj:, n] = weights[i_traj, n - 1]
                break
            if soften:
                weights[i_traj, n] = traj_set.trajectories[i_traj][n].acc_soft_isweight[0].item()
            else:
                # あるtrajectory i_traj における step n においてのimportance samplingをweightに保存する
                weights[i_traj, n] = traj_set.trajectories[i_traj][n].acc_isweight[0].item()

    # WISの場合にはtrajectoryで得られた重みを合計した値で重みを割る
    if wis:
        for n in range(config.max_length):
            weights_sum[n] = np.sum(weights[:, n])
            if weights_sum[n] != 0:
                weights[:, n] = (weights[:, n] * num_samples) / weights_sum[n]

    value = np.zeros(num_samples)
    for i_traj in range(num_samples):
        w = 1
        imd_rwd = 0
        discount_factor = 1
        discount_rate = 1

        imd_rwd_list = []
        discount_factor_list = []
        gamma_mu_list = []
        gamma_q_list = []
        gamma_v_list = []
        value_right_list = []

        for t in traj_set.trajectories[i_traj]:

            imd_rwd = imd_rwd + t.reward[0].item() * discount_factor

            gamma_mu = mu_hat[i_traj, t.time] * discount_factor
            gamma_q = Q_value[i_traj, t.time] * discount_factor
            gamma_v = V_value[i_traj, t.time] * discount_factor

            imd_rwd_list.append(imd_rwd)
            discount_factor_list.append(discount_factor)
            gamma_mu_list.append(gamma_mu)
            gamma_q_list.append(gamma_q)
            gamma_v_list.append(gamma_v)

            if t.time == 0:
                value_right = weights[i_traj, t.time] * gamma_q - gamma_v
            else:
                value_right = weights[i_traj, t.time] * (np.sum(gamma_mu_list[:t.time] + gamma_q_list)) - weights[
                    i_traj, np.int(t.time - 1)] * (np.sum(gamma_mu_list[:t.time] + gamma_v))

            value_right_list.append(value_right)
            discount_factor = discount_factor ** discount_rate

        value_left = weights[i_traj, t.time] * np.sum(imd_rwd_list)
        value[i_traj] = (value_left - np.sum(value_right))

    return value


def dml_doubly_robust(traj_set, V_value, Q_value, pz, config, wis=False, soften=False):
    num_samples = len(traj_set)
    weights = np.zeros((num_samples, config.max_length))
    weights_sum = np.zeros(config.max_length)

    for i_traj in range(num_samples):
        acc_pz = 1
        for n in range(config.max_length):

            if n >= len(traj_set.trajectories[i_traj]):  # trajectoryの長さを超えたらbreak
                weights[i_traj:, n] = weights[i_traj, n - 1]
                break

            acc_w = traj_set.trajectories[i_traj][n].acc_isweight[0].item()
            action = traj_set.trajectories[i_traj][n].action[0].item()
            eval_policy = traj_set.trajectories[i_traj][n].pie[0]
            est_pizero = pz[i_traj][n].item()

            iw = eval_policy[action].item() / est_pizero
            acc_pz = acc_pz * iw
            # print(n, eval_policy[action].item(), est_pizero, iw, acc_pz, acc_w)
            weights[i_traj, n] = acc_pz

    # WISの場合にはtrajectoryで得られた重みを合計した値で重みを割る
    if wis:
        for n in range(config.max_length):
            weights_sum[n] = np.sum(weights[:, n])
            if weights_sum[n] != 0:
                weights[:, n] = (weights[:, n] * num_samples) / weights_sum[n]

    value = np.zeros(num_samples)
    for i_traj in range(num_samples):
        w = 1
        for t in traj_set.trajectories[i_traj]:
            # trajectory i_traj のstep tでVを計算する
            # t.reward[0].item() : ログ上の報酬
            # Q_value[i_traj,t.time] : Qの予測値
            # V_value[i_traj,t.time] : Vの予測値
            value[i_traj] += weights[i_traj, t.time] * (t.reward[0].item() - Q_value[i_traj, t.time]) + w * V_value[
                i_traj, t.time]
            w = weights[i_traj, t.time]
            if w == 0:
                break
    return value


def importance_sampling(traj_set, wis=False, soften=False):
    num_samples = len(traj_set)
    value = np.zeros(num_samples)
    weights = np.zeros(num_samples)

    for i_traj in range(num_samples):
        l = len(traj_set.trajectories[i_traj])
        if soften:
            weights[i_traj] = traj_set.trajectories[i_traj][l - 1].acc_soft_isweight[0].item()
        else:
            weights[i_traj] = traj_set.trajectories[i_traj][l - 1].acc_isweight[0].item()

    if wis:
        weights = (weights * num_samples) / np.sum(weights)

    for i_traj in range(num_samples):
        l = len(traj_set.trajectories[i_traj])
        value[i_traj] = l * weights[i_traj]

    return value


def mrdr_preprocess(traj_set, config):
    transitions = []
    weights = np.zeros((len(traj_set), config.max_length, config.max_length))
    weights_sum = np.zeros((config.max_length, config.max_length))
    weights_num = np.zeros((config.max_length, config.max_length))
    for i in range(len(traj_set)):  # trajectory
        for n1 in range(config.max_length):
            if n1 >= len(traj_set.trajectories[i]):  # i番目のtrajの長さよりも n1が大きい場合
                weights[i, n1:, :] = 0  # weightを0にする
                break
            for n2 in range(n1, config.max_length):
                if n2 >= len(traj_set.trajectories[i]):
                    weights[i, n1, n2:] = weights[i, n1, n2 - 1]
                    weights_sum[n1, n2:] += weights[i, n1, n2 - 1]
                    weights_num[n1, n2:] += 1
                    break
                weights[i, n1, n2] = traj_set.trajectories[i][n2].acc_soft_isweight \
                                     / traj_set.trajectories[i][n1].acc_soft_isweight
                weights_sum[n1, n2] += weights[i, n1, n2]
                weights_num[n1, n2] += 1

    for i in range(len(traj_set)):
        traj = traj_set.trajectories[i]
        for n1 in range(len(traj)):
            R = 0
            WR = 0
            for n2 in range(n1, len(traj)):
                R += traj[n2].reward * weights[i, n1, n2]
                WR += traj[n2].reward * weights[i, n1, n2] * weights_num[n1, n2] / weights_sum[n1, n2]
                # print(weights[i,n1,n2], weights[i,n1,n2]*weights_num[n1,n2]/weights_sum[n1,n2])
                # print(n1, n2, weights_num[n1,n2], weights_sum[n1,n2])
            transitions.append(MRDRTransition(*traj[n1], R, WR))
            # print(n1, R[0], WR[0])
    return transitions


def train_pipeline(env, config, eval_qnet, bhv_qnet, seedvec=None):
    memory = SampleSet(config)  # same the tuples for model training
    dev_memory = SampleSet(config)
    pz_memory = SampleSet(config)

    noise_dim = config.noise_dim
    state_dim = config.state_dim + noise_dim

    fold_num = config.fold_num
    rep_memory_k = [SampleSet(config) for i in range(fold_num)]
    rep_dev_memory_k = [SampleSet(config) for i in range(fold_num)]

    nd_memory_k = [SampleSet(config) for i in range(fold_num)]
    nd_memory_test = [SampleSet(config) for i in range(fold_num)]

    fold_sample_num = int(np.trunc(config.sample_num_traj / fold_num))

    traj_set = TrajectorySet(config)  # save the trajectory for doubly robust evaluation
    scores = deque()
    mdpnet_msepi = MDPnet(config)
    mdpnet = MDPnet(config)  # doubly robustで使う
    mdpnet_unweight = MDPnet(config)

    mrdr_q = QtNet(state_dim, config.mrdr_hidden_dims, config.action_size)
    mrdrv2_q = QtNet(state_dim, config.mrdr_hidden_dims, config.action_size)
    mdpnet_dml = Policynet(config)
    time_pre = time.time()

    print(config)

    # fix initial state
    if seedvec is None:
        seedvec = np.random.randint(0, config.MAX_SEED, config.sample_num_traj_eval)

    # 同じQ functionをつかって、
    # greedyにやるのがevaluation policy
    # epsilon greedyにやるのがbehavior policy
    for i_episode in range(config.sample_num_traj):  # 評価trajectoryの作成
        # Initialize the environment and state
        randseed = seedvec[i_episode].item()
        env.seed(randseed)

        state = np.append(env.reset(), np.random.normal(size = noise_dim))
        state = preprocess_state(state, state_dim)  # stateを行列にしてtensor形式にする i_episodeごとに環境をリセットしている

        done = False
        n_steps = 0
        acc_soft_isweight = FloatTensor([1])
        acc_isweight = FloatTensor([1])
        factual = 1
        last_factual = 1
        traj_set.new_traj()  # traj_setに新しいリストを追加

        while not done:
            # Select and perform an action
            q_values = bhv_qnet.forward(state.type(Tensor)).detach()  # Q値の予測 detachは勾配が伝わらないようにする処理
            # ここのactionはbehavior policyによるもの？
            # q-valuesはNULLじゃないからqnetは使われない
            # actionは結局trajectoryとして保存されるのでbehaviorっぽい
            action = epsilon_greedy_action(state, bhv_qnet, config.behavior_epsilon, config.action_size,
                                           q_values)  # epsilon greedyでactionを決定する(behavior)
            p_pib = epsilon_greedy_action_prob(state, bhv_qnet, config.behavior_epsilon,
                                               config.action_size,
                                               q_values)  # epsilon greedy (behavior policy) return list of probability
            soft_pie = epsilon_greedy_action_prob(state, bhv_qnet, config.soften_epsilon,
                                                  config.action_size,
                                                  q_values)  # soften_epsilonを使った時の epsilon greedyの各腕の選択確率を返す
            p_pie = epsilon_greedy_action_prob(state, bhv_qnet, 0, config.action_size,
                                               q_values)  # argmax(evaluation policy)
            print

            isweight = p_pie[:, action.item()] / p_pib[:, action.item()]  # importance weightの算出 max選択肢以外は０
            acc_isweight = acc_isweight * (p_pie[:, action.item()] / p_pib[:, action.item()])
            soft_isweight = (soft_pie[:, action.item()] / p_pib[:, action.item()])  # soften_epsilonを使った時の IW
            acc_soft_isweight = acc_soft_isweight * (soft_pie[:, action.item()] / p_pib[:, action.item()])

            last_factual = factual * (1 - p_pie[:, action.item()])  # 1{a_{0:t-1}==\pie, a_t != \pie}
            factual = factual * p_pie[:, action.item()]  # 1{a_{0:t}==\pie}

            next_state, reward, done, _ = env.step(action.item())  # action の実行 (doneはここで更新される）
            next_state = np.append(next_state, np.random.normal(size = noise_dim))

            reward = Tensor([reward])  # 報酬をtensor形式にしておく
            next_state = preprocess_state(next_state, state_dim)  # 次のstateの保存形式を変更
            next_state_re = torch.tensor(np.float32(config.rescale)) * next_state  # stateをrescaleする
            state_re = torch.tensor(np.float32(config.rescale)) * state  # stateをrescaleする

            # train数に指定された数以下であればDR等のmdp modelの学習のためのデータをmemoryに保存しておく
            # それ以上の場合にはdev_memoryに保存しておく
            if i_episode < config.train_num_traj:
                memory.push(state_re, action, next_state_re, reward, done, isweight, acc_isweight, n_steps, factual,
                            last_factual, acc_soft_isweight, soft_isweight, soft_pie, p_pie, p_pib)
            else:
                dev_memory.push(state_re, action, next_state_re, reward, done, isweight, acc_isweight, n_steps, factual,
                                last_factual, acc_soft_isweight, soft_isweight, soft_pie, p_pie, p_pib)

            # cross fitting memory for k fold
            fold = int(np.trunc(i_episode / fold_sample_num))
            fold_idx_list = np.arange(fold_num)

            if fold >= np.max(fold_idx_list):
                dev_fold = 0
            else:
                dev_fold = fold + 1

            # fordと一致しないkの身を残す
            fold_idx_list_k = [x for x in fold_idx_list if x not in [fold]]

            for memory_idx in fold_idx_list_k:
                nd_memory_k[memory_idx].push(state_re, action, next_state_re, reward, done, isweight, acc_isweight,
                                             n_steps, factual,
                                             last_factual, acc_soft_isweight, soft_isweight, soft_pie, p_pie, p_pib)
            # target fold data save
            nd_memory_test[fold].push(state_re, action, next_state_re, reward, done, isweight, acc_isweight, n_steps,
                                      factual,
                                      last_factual, acc_soft_isweight, soft_isweight, soft_pie, p_pie, p_pib)

            # fold, dev_foldと一致しないkのみを残す
            # fold_idx_list_k = [x for x in fold_idx_list if x not in [fold, dev_fold]]

            # fold, dev_foldでもない全てのmemoryにログを残す
            for memory_idx in fold_idx_list_k:
                rep_memory_k[memory_idx].push(state_re, action, next_state_re, reward, done, isweight, acc_isweight,
                                              n_steps, factual,
                                              last_factual, acc_soft_isweight, soft_isweight, soft_pie, p_pie, p_pib)

            # dev_foldのmemoryにログを残す
            rep_dev_memory_k[dev_fold].push(state_re, action, next_state_re, reward, done, isweight, acc_isweight,
                                            n_steps, factual,
                                            last_factual, acc_soft_isweight, soft_isweight, soft_pie, p_pie, p_pib)

            # pizero用のmemory(memory + dev_memoryになる)
            pz_memory.push(state_re, action, next_state_re, reward, done, isweight, acc_isweight, n_steps, factual,
                           last_factual, acc_soft_isweight, soft_isweight, soft_pie, p_pie, p_pib)

            # OPEに利用するデータ
            traj_set.push(state, action, next_state, reward, done, isweight, acc_isweight, n_steps, factual,
                          last_factual, acc_soft_isweight, soft_isweight, soft_pie, p_pie, p_pib)

            state = FloatTensor(next_state)
            n_steps += 1

        scores.append(n_steps)
    memory.flatten()  # prepare flatten data
    dev_memory.flatten()

    memory.update_u()  # prepare u_{0:t}
    dev_memory.update_u()

    mean_score = np.mean(scores)
    print('Sampling {} trajectories, the mean survival time is {}'
          .format(config.sample_num_traj, mean_score))
    print('{} train samples, {} dev sample'.format(len(memory), len(dev_memory)))
    # print(len(memory.terminal))
    time_now = time.time()
    time_sampling = time_now - time_pre
    time_pre = time_now

    mrdr_samples = mrdr_preprocess(traj_set, config)  # 保存したtrajectoriesをmrdrの学習でーたに変換する
    total_num = len(mrdr_samples)
    mrdr_train_samples = mrdr_samples[: int(total_num * 0.9)]  # 90%のデータを学習に使う
    mrdr_dev_samples = mrdr_samples[int(total_num * 0.9):]
    time_now = time.time()
    time_premrdr = time_now - time_pre
    time_pre = time_now

    # print('Learn MRDR Q functions')
    # best_train_loss = 10000
    # lr = config.mrdr_lr # learning rate?
    # optimizer = optim.Adam(mrdr_q.parameters(), lr=lr)
    # for i_episode in range(config.mrdr_num_episodes):
    #     train_loss = 0
    #     dev_loss = 0
    #     train_permutation = np.random.permutation(len(mrdr_train_samples)) #順番をシャッフル？
    #     dev_permutation = np.random.permutation(len(mrdr_dev_samples))
    #
    #     for i_batch in range(config.mrdr_num_batches):
    #         train_loss_batch = mrdr_train(mrdr_train_samples, mrdr_q, optimizer,
    #                                       'd', config, train_permutation, i_batch, wis_reward = False)
    #         dev_loss_batch = mrdr_test(mrdr_dev_samples, mrdr_q,
    #                                    'd', config, dev_permutation, i_batch, wis_reward = False)
    #         train_loss = (train_loss * i_batch + train_loss_batch) / (i_batch + 1)
    #         dev_loss = (dev_loss * i_batch + dev_loss_batch) / (i_batch + 1)
    #
    #     if (i_episode + 1) % config.print_per_epi == 0:
    #         print('Episode {:0>3d}: train loss {:.3e}, dev loss {:.3e}, lr={}'
    #               .format(i_episode + 1, train_loss, dev_loss, lr))
    #
    #     if train_loss < best_train_loss:
    #         best_train_loss = train_loss
    #     else:
    #         lr *= 0.9
    #         learning_rate_update(optimizer, lr)
    #
    # time_now = time.time()
    # time_mrdr = time_now - time_pre
    # time_pre = time_now

    # print('Learn MRDR-WIS Q functions')
    # best_train_loss = 10000
    # lr = config.mrdr_lr
    # optimizer = optim.Adam(mrdrv2_q.parameters(), lr=lr)
    # for i_episode in range(config.mrdr_num_episodes):
    #     train_loss = 0
    #     dev_loss = 0
    #     train_permutation = np.random.permutation(len(mrdr_train_samples))
    #     dev_permutation = np.random.permutation(len(mrdr_dev_samples))
    #     for i_batch in range(config.mrdr_num_batches):
    #         train_loss_batch = mrdr_train(mrdr_train_samples, mrdrv2_q, optimizer,
    #                                       'd', config, train_permutation, i_batch, wis_reward=True)
    #         dev_loss_batch = mrdr_test(mrdr_dev_samples, mrdrv2_q,
    #                                    'd', config, dev_permutation, i_batch, wis_reward=True)
    #         train_loss = (train_loss * i_batch + train_loss_batch) / (i_batch + 1)
    #         dev_loss = (dev_loss * i_batch + dev_loss_batch) / (i_batch + 1)
    #     if (i_episode + 1) % config.print_per_epi == 0:
    #         print('Episode {:0>3d}: train loss {:.3e}, dev loss {:.3e}, lr={}'
    #               .format(i_episode + 1, train_loss, dev_loss, lr))
    #     if train_loss < best_train_loss:
    #         best_train_loss = train_loss
    #     else:
    #         lr *= 0.9
    #         learning_rate_update(optimizer, lr)
    # time_now = time.time()
    # time_mrdr += time_now - time_pre
    # time_pre = time_now

    print('Learn mse_pi mdp model')
    best_train_loss = 100
    lr = config.lr
    for i_episode in range(config.train_num_episodes):
        train_loss = 0
        dev_loss = 0
        optimizer = optim.Adam(mdpnet_msepi.parameters(), lr=lr, weight_decay=config.weight_decay)
        for i_batch in range(config.train_num_batches):
            train_loss_batch = mdpmodel_train(memory, mdpnet_msepi, optimizer, 2, config)
            dev_loss_batch = mdpmodel_test(dev_memory, mdpnet_msepi, 2, config)

            train_loss = (train_loss * i_batch + train_loss_batch) / (i_batch + 1)
            dev_loss = (dev_loss * i_batch + dev_loss_batch) / (i_batch + 1)
        if (i_episode + 1) % config.print_per_epi == 0:
            print('Episode {:0>3d}: train loss {:.3e}, dev loss {:.3e}'
                  .format(i_episode + 1, train_loss, dev_loss))
        if train_loss < best_train_loss:
            best_train_loss = train_loss
        else:
            lr *= config.lr_decay

    print('Learn pizero model')
    best_train_loss = 100
    lr = config.lr
    pizero_episode = 500
    dev_loss_vec = np.zeros(pizero_episode)
    for i_episode in range(pizero_episode):
        train_loss = 0
        dev_loss = 0
        optimizer = optim.SGD(mdpnet_dml.parameters(), lr=config.policy_lr, weight_decay=config.weight_decay)
        for i_batch in range(config.policy_train_num_batches):
            train_loss_batch = pzmodel_train(memory, mdpnet_dml, optimizer, 3, config)
            dev_loss_batch = pzmodel_test(dev_memory, mdpnet_dml, 3, config)


            train_loss = (train_loss * i_batch + train_loss_batch) / (i_batch + 1)
            dev_loss = (dev_loss * i_batch + dev_loss_batch) / (i_batch + 1)

        if (i_episode + 1) % config.print_per_epi == 0:
            print('Episode {:0>3d}: train loss {:.3e}, dev loss {:.3e}, best loss {:.3e}'
                  .format(i_episode + 1, train_loss, dev_loss, best_train_loss))

        dev_loss_vec[i_episode] = dev_loss

        if train_loss < best_train_loss:
            best_train_loss = train_loss
        else:
            lr *= config.lr_decay

        # if (i_episode > 20) & (dev_loss_vec[i_episode - 20] < dev_loss):
        #     best_train_loss = train_loss
        #     break
        # else:
        #     lr *= config.lr_decay

    # print('Learn RepBM mdp model')
    # best_train_loss = 100
    # lr = config.lr
    # for i_episode in range(config.train_num_episodes):
    #     train_loss = 0
    #     dev_loss = 0
    #     optimizer = optim.Adam(mdpnet.parameters(), lr=lr, weight_decay=config.weight_decay)
    #     for i_batch in range(config.train_num_batches):
    #         train_loss_batch = mdpmodel_train(memory, mdpnet, optimizer, 1, config)
    #         dev_loss_batch = mdpmodel_test(dev_memory, mdpnet, 1, config)
    #         train_loss = (train_loss * i_batch + train_loss_batch) / (i_batch + 1)
    #         dev_loss = (dev_loss * i_batch + dev_loss_batch) / (i_batch + 1)
    #     if (i_episode + 1) % config.print_per_epi == 0:
    #         print('Episode {:0>3d}: train loss {:.3e}, dev loss {:.3e}'
    #               .format(i_episode + 1, train_loss, dev_loss))
    #     if train_loss < best_train_loss:
    #         best_train_loss = train_loss
    #     else:
    #         lr *= config.lr_decay

    print('Learn the baseline mdp model')
    best_train_loss = 100
    lr = config.lr
    for i_episode in range(config.train_num_episodes):
        train_loss = 0
        dev_loss = 0
        optimizer = optim.Adam(mdpnet_unweight.parameters(), lr=lr, weight_decay=config.weight_decay)
        for i_batch in range(config.train_num_batches):
            train_loss_batch = mdpmodel_train(memory, mdpnet_unweight, optimizer, 0, config)
            train_loss = (train_loss * i_batch + train_loss_batch) / (i_batch + 1)
            dev_loss_batch = mdpmodel_test(dev_memory, mdpnet_unweight, 0, config)
            dev_loss = (dev_loss * i_batch + dev_loss_batch) / (i_batch + 1)
        if (i_episode + 1) % config.print_per_epi == 0:
            print('Episode {:0>3d}: train loss {:.3e}, dev loss {:.3e}, best loss {:.3e}'
                  .format(i_episode + 1, train_loss, dev_loss, best_train_loss))
        if dev_loss < best_train_loss:
            best_train_loss = dev_loss
        else:
            lr *= config.lr_decay
    time_now = time.time()
    time_mdp = time_now - time_pre
    time_pre = time_now

    tc = TerminalClassifier(config)
    print('Learn terminal classifier')
    lr = 0.01
    best_train_acc = 0
    optimizer = optim.Adam([param for param in tc.parameters() if param.requires_grad], lr=lr)
    for i_episode in range(config.tc_num_episode):
        train_loss = 0
        dev_loss = 0
        train_acc = 0
        dev_acc = 0
        for i_batch in range(config.tc_num_batches):
            train_loss_batch, train_acc_batch = terminal_classifier_train(memory, tc, optimizer,
                                                                          config.tc_batch_size)
            dev_loss_batch, dev_acc_batch = terminal_classifier_test(dev_memory, tc,
                                                                     config.tc_test_batch_size)
            train_loss = (train_loss * i_batch + train_loss_batch) / (i_batch + 1)
            train_acc = (train_acc * i_batch + train_acc_batch) / (i_batch + 1)
            dev_loss = (dev_loss * i_batch + dev_loss_batch) / (i_batch + 1)
            dev_acc = (dev_acc * i_batch + dev_acc_batch) / (i_batch + 1)
        if (i_episode + 1) % config.print_per_epi == 0:
            print('Episode {:0>3d}: train loss {:.3e} acc {:.3e}, dev loss {:.3e} acc {:.3e}'.
                  format(i_episode + 1, train_loss, train_acc, dev_loss, dev_acc))
        if train_acc > best_train_acc:
            best_train_acc = train_acc
        else:
            lr *= 0.9
            learning_rate_update(optimizer, lr)
    time_now = time.time()
    time_tc = time_now - time_pre
    time_pre = time_now

    print('Evaluate models using evaluation policy on the same initial states')
    mdpnet.eval()
    mdpnet_unweight.eval()
    mdpnet_msepi.eval()
    target = np.zeros(config.sample_num_traj)

    init_states = []
    for i_episode in range(config.sample_num_traj):
        env.seed(seedvec[i_episode].item())
        state = env.reset()
        state = np.append(state, np.random.normal(size = noise_dim))
        init_states.append(preprocess_state(state, state_dim))
    init_states = torch.cat(init_states)

    # RepBM model with representation loss
    # mv = rollout_batch(init_states, mdpnet, tc, config.eval_num_rollout, eval_qnet,
    #                   epsilon=0, action_size=config.action_size, maxlength=config.max_length, config=config)
    # Usual MDP model
    mv_bsl = rollout_batch(init_states, mdpnet_unweight, tc, config.eval_num_rollout, eval_qnet,
                           epsilon=0, action_size=config.action_size, maxlength=config.max_length, config=config)
    # RepBM model without representation loss
    # mv_msepi = rollout_batch(init_states, mdpnet_msepi, tc, config.eval_num_rollout, eval_qnet,
    #                    epsilon=0, action_size=config.action_size, maxlength=config.max_length, config=config)
    time_now = time.time()
    time_eval = time_now - time_pre
    time_pre = time_now

    # -----------------
    print('pizero estimation')
    pz = compute_pizero(traj_set, mdpnet_dml, tc, eval_qnet, config)

    # -----------------

    # k fold cross fitting no dev
    dml_dr_cross_k_nd = np.zeros(config.sample_num_traj)
    dml_dr_cross_k_estpz_nd = np.zeros(config.sample_num_traj)
    dml_dr_cross_k_estpz_wis_nd = np.zeros(config.sample_num_traj)
    dml_dr_cross_k_chunk_nd = np.zeros(config.sample_num_traj)
    dml_dr_cross_k_estpz_sis_nd = np.zeros(config.sample_num_traj)
    dml_dr_cross_k_estpz_swis_nd = np.zeros(config.sample_num_traj)

    dml_dr_cross_k_nd_repbm = np.zeros(config.sample_num_traj)
    dml_dr_cross_k_estpz_nd_repbm = np.zeros(config.sample_num_traj)
    dml_dr_cross_k_repbm_estpz_wis_nd = np.zeros(config.sample_num_traj)
    dml_dr_cross_k_repbm_estpz_sis_nd = np.zeros(config.sample_num_traj)
    dml_dr_cross_k_repbm_estpz_swis_nd = np.zeros(config.sample_num_traj)

    fold_indicator = np.trunc(np.arange(config.sample_num_traj) / fold_sample_num)
    for fold_idx in range(fold_num):

        if fold_idx >= (fold_num - 1):
            dev_fold = 0
        else:
            dev_fold = fold_idx + 1

        # prepare memory
        fold_memory = nd_memory_k[fold_idx]
        fold_memory.flatten()
        fold_memory.update_u()

        fold_dev_memory = nd_memory_test[dev_fold]
        fold_dev_memory.flatten()
        fold_dev_memory.update_u()

        rep_fold_memory = rep_memory_k[fold_idx]
        rep_fold_memory.flatten()
        rep_fold_memory.update_u()

        rep_dev_memory = rep_dev_memory_k[fold_idx]
        rep_dev_memory.flatten()
        rep_dev_memory.update_u()

        # learning rate setting
        best_train_loss = 100
        lr = config.lr

        mdpnet_dml_crossfit_k = MDPnet(config)

        for i_episode in range(config.train_num_episodes):  # range(len(memory_k_two)):  # len(cross_idx)
            train_loss = 0
            dev_loss = 0
            optimizer_one = optim.Adam(mdpnet_dml_crossfit_k.parameters(), lr=lr, weight_decay=config.weight_decay)
            for i_batch in range(config.train_num_batches):
                train_loss_batch = mdpmodel_train(fold_memory, mdpnet_dml_crossfit_k, optimizer_one, 0, config)
                train_loss = (train_loss * i_batch + train_loss_batch) / (i_batch + 1)
                dev_loss_batch = mdpmodel_test(fold_dev_memory, mdpnet_dml_crossfit_k, 0, config)
                dev_loss = (dev_loss * i_batch + dev_loss_batch) / (i_batch + 1)

            if (i_episode + 1) % config.print_per_epi == 0:
                print('Episode {:0>3d}: train loss {:.3e}, dev loss {:.3e}, best loss {:.3e}'
                      .format(i_episode + 1, train_loss, dev_loss, best_train_loss))

            if dev_loss < best_train_loss:  # if train_loss < best_train_loss:
                best_train_loss = dev_loss  # best_train_loss = train_loss
            else:
                lr *= config.lr_decay

        mdpnet_dml_crossfit_k.eval()

        print("eval dr_dml_cross fold " + str(fold_idx))

        # calculate V,Q for fold = k
        V, Q = compute_values_dml(traj_set, mdpnet_dml_crossfit_k, tc, eval_qnet, config,
                                  max_length=config.max_length,
                                  model_type='MDP')

        # reward predictin
        mu_hat = reward_prediction(traj_set, mdpnet_dml_crossfit_k, max_length=config.max_length, config=config,
                                   soften=False)

        # doubly robust estimate for fold = k
        dml_dr_cross_k_fold = doubly_robust(traj_set, V, Q, config, wis=False, soften=False)
        dml_dr_cross_k_nd[fold_indicator == fold_idx] = dml_dr_cross_k_fold[fold_indicator == fold_idx]

        # doubly robust with estimated ps for fold = k
        dml_dr_cross_estpz_k_fold = dml_doubly_robust(traj_set, V, Q, pz, config, wis=False, soften=False)
        dml_dr_cross_k_estpz_nd[fold_indicator == fold_idx] = dml_dr_cross_estpz_k_fold[fold_indicator == fold_idx]

        # doubly robust with self normalized estimated ps for fold = k
        dml_dr_cross_estpz_k_fold_wis = dml_doubly_robust(traj_set, V, Q, pz, config, wis=True, soften=False)
        dml_dr_cross_k_estpz_wis_nd[fold_indicator == fold_idx] = dml_dr_cross_estpz_k_fold_wis[
            fold_indicator == fold_idx]

        # doubly robust with self normalized estimated ps for fold = k
        dml_dr_cross_estpz_k_fold_sis = dml_doubly_robust(traj_set, V, Q, pz, config, wis=False, soften=True)
        dml_dr_cross_k_estpz_sis_nd[fold_indicator == fold_idx] = dml_dr_cross_estpz_k_fold_sis[
            fold_indicator == fold_idx]

        # doubly robust with self normalized estimated ps for fold = k
        dml_dr_cross_estpz_k_fold_swis = dml_doubly_robust(traj_set, V, Q, pz, config, wis=True, soften=True)
        dml_dr_cross_k_estpz_swis_nd[fold_indicator == fold_idx] = dml_dr_cross_estpz_k_fold_swis[
            fold_indicator == fold_idx]

        # doubly robust estimate for fold = k (chunk)
        dml_dr_cross_k_fold_chunk_nd = doubly_robust_dml_chunk(traj_set, V, Q, mu_hat, config, wis=False, soften=False)
        dml_dr_cross_k_chunk_nd[fold_indicator == fold_idx] = dml_dr_cross_k_fold_chunk_nd[fold_indicator == fold_idx]
        del mdpnet_dml_crossfit_k
        del V
        del Q

        ##

        # # learning rate setting
        # best_train_loss = 100
        # lr = config.lr
        #
        # #model prepare
        # mdpnet_dml_repbm_crossfit_k = MDPnet(config)
        #
        # #learn model
        # for i_episode in range(100):  # range(len(memory_k_two)):  # len(cross_idx)
        #     train_loss = 0
        #     dev_loss = 0
        #     optimizer_one = optim.Adam(mdpnet_dml_repbm_crossfit_k.parameters(), lr=lr, weight_decay=config.weight_decay)
        #     for i_batch in range(config.train_num_batches):
        #         train_loss_batch = mdpmodel_train(rep_fold_memory, mdpnet_dml_repbm_crossfit_k, optimizer_one, 1, config)
        #         train_loss = (train_loss * i_batch + train_loss_batch) / (i_batch + 1)
        #         dev_loss_batch = mdpmodel_test(rep_dev_memory, mdpnet_dml_repbm_crossfit_k, 1, config)
        #         dev_loss = (dev_loss * i_batch + dev_loss_batch) / (i_batch + 1)
        #
        #     if (i_episode + 1) % config.print_per_epi == 0:
        #         print('Episode {:0>3d}: train loss {:.3e}, dev loss {:.3e}'.format(i_episode + 1, train_loss,
        #                                                                            dev_loss))
        #
        #     if train_loss < best_train_loss:
        #         best_train_loss = train_loss
        #     else:
        #         lr *= config.lr_decay
        #
        # mdpnet_dml_repbm_crossfit_k.eval()
        #
        # # calculate V,Q for fold = k(RepBM DML)
        # V, Q = compute_values_dml(traj_set, mdpnet_dml_repbm_crossfit_k, tc, eval_qnet, config,
        #                           max_length=config.max_length,
        #                           model_type='MDP')
        #
        # # doubly robust estimate for fold = k
        # dml_dr_cross_k_nd_repbm_fold = doubly_robust(traj_set, V, Q, config, wis=False, soften=False)
        # dml_dr_cross_k_nd_repbm[fold_indicator == fold_idx] = dml_dr_cross_k_nd_repbm_fold[fold_indicator == fold_idx]
        #
        # # doubly robust with estimated ps for fold = k
        # dml_dr_cross_k_estpz_nd_repbm_fold = dml_doubly_robust(traj_set, V, Q, pz, config, wis=False, soften=False)
        # dml_dr_cross_k_estpz_nd_repbm[fold_indicator == fold_idx] = dml_dr_cross_k_estpz_nd_repbm_fold[fold_indicator == fold_idx]
        #
        # # doubly robust with self normalized estimated ps for fold = k
        # dml_dr_cross_estpz_k_repbm_fold_wis = dml_doubly_robust(traj_set, V, Q, pz, config, wis=True, soften=False)
        # dml_dr_cross_k_repbm_estpz_wis_nd[fold_indicator == fold_idx] = dml_dr_cross_estpz_k_repbm_fold_wis[fold_indicator == fold_idx]
        #
        # # doubly robust with self normalized estimated ps for fold = k
        # dml_dr_cross_estpz_k_repbm_fold_sis = dml_doubly_robust(traj_set, V, Q, pz, config, wis=False, soften=True)
        # dml_dr_cross_k_repbm_estpz_sis_nd[fold_indicator == fold_idx] = dml_dr_cross_estpz_k_repbm_fold_sis[fold_indicator == fold_idx]
        #
        # # doubly robust with self normalized estimated ps for fold = k
        # dml_dr_cross_estpz_k_repbm_fold_swis = dml_doubly_robust(traj_set, V, Q, pz, config, wis=True, soften=True)
        # dml_dr_cross_k_repbm_estpz_swis_nd[fold_indicator == fold_idx] = dml_dr_cross_estpz_k_repbm_fold_swis[fold_indicator == fold_idx]
        #
        # del mdpnet_dml_repbm_crossfit_k
        # del V
        # del Q
    # -----------------

    # RepBM(proposed method)
    print("eval RepBM")  # RepBM without representation
    # V,Q = compute_values(traj_set, mdpnet_msepi, tc, eval_qnet, config, max_length = config.max_length, model_type='MDP')
    # dr_msepi = doubly_robust(traj_set, V, Q, config, wis=False, soften=False)
    # wdr_msepi = doubly_robust(traj_set, V, Q, config, wis=True, soften=False)
    # sdr_msepi = doubly_robust(traj_set, V, Q, config, wis=False, soften=True)
    # swdr_msepi = doubly_robust(traj_set, V, Q, config, wis=True, soften=True)
    # dr_msepi_estpz = dml_doubly_robust(traj_set, V, Q, pz, config, wis=False, soften=False)
    # wdr_msepi_estpz = dml_doubly_robust(traj_set, V, Q, pz, config, wis=True, soften=False)
    # sdr_msepi_estpz = dml_doubly_robust(traj_set, V, Q, pz, config, wis=False, soften=True)
    # swdr_msepi_estpz = dml_doubly_robust(traj_set, V, Q, pz, config, wis=True, soften=True)

    # RepBM with representation
    # V,Q = compute_values(traj_set, mdpnet, tc, eval_qnet, config, max_length = config.max_length, model_type='MDP')
    # dr = doubly_robust(traj_set, V, Q, config, wis=False, soften=False)
    # wdr = doubly_robust(traj_set, V, Q, config, wis=True, soften=False)
    # sdr = doubly_robust(traj_set, V, Q, config, wis=False, soften=True)
    # swdr = doubly_robust(traj_set, V, Q, config, wis=True, soften=True)
    # dr_estpz = dml_doubly_robust(traj_set, V, Q, pz, config, wis=False, soften=False)
    # wdr_estpz = dml_doubly_robust(traj_set, V, Q, pz, config, wis=True, soften=False)
    # sdr_estpz = dml_doubly_robust(traj_set, V, Q, pz, config, wis=False, soften=True)
    # swdr_estpz = dml_doubly_robust(traj_set, V, Q, pz, config, wis=True, soften=True)

    print("eval dr_bsl")
    V, Q = compute_values(traj_set, mdpnet_unweight, tc, eval_qnet, config, max_length=config.max_length,
                          model_type='MDP')
    dr_bsl = doubly_robust(traj_set, V, Q, config, wis=False, soften=False)
    wdr_bsl = doubly_robust(traj_set, V, Q, config, wis=True, soften=False)
    sdr_bsl = doubly_robust(traj_set, V, Q, config, wis=False, soften=True)
    swdr_bsl = doubly_robust(traj_set, V, Q, config, wis=True, soften=True)

    dr_bsl_estpz = dml_doubly_robust(traj_set, V, Q, pz, config, wis=False, soften=False)
    wdr_bsl_estpz = dml_doubly_robust(traj_set, V, Q, pz, config, wis=True, soften=False)

    print("eval MoreRobust DR")
    # V, Q = compute_values(traj_set, mrdr_q, tc, eval_qnet, config, max_length = config.max_length, model_type='Q')
    # mrdr_qv = V[:, 0]
    # mrdr = doubly_robust(traj_set, V, Q, config, wis=False, soften=False)
    # wmrdr = doubly_robust(traj_set, V, Q, config, wis=True, soften=False)
    #
    # V, Q = compute_values(traj_set, mrdr_q, tc, eval_qnet, config, max_length = config.max_length, model_type='Q', soften=True)
    # smrdr = doubly_robust(traj_set, V, Q, config, wis=False, soften=True)
    # swmrdr = doubly_robust(traj_set, V, Q, config, wis=True, soften=True)
    #
    # V, Q = compute_values(traj_set, mrdrv2_q, tc, eval_qnet, config, max_length = config.max_length, model_type='Q')
    # mrdrv2_qv = V[:, 0]
    # mrdrv2 = doubly_robust(traj_set, V, Q, config, wis=False, soften=False)
    # wmrdrv2 = doubly_robust(traj_set, V, Q, config, wis=True, soften=False)
    # V, Q = compute_values(traj_set, mrdrv2_q, tc, eval_qnet, config, max_length = config.max_length, model_type='Q', soften=True)
    # smrdrv2 = doubly_robust(traj_set, V, Q, config, wis=False, soften=True)
    # swmrdrv2 = doubly_robust(traj_set, V, Q, config, wis=True, soften=True)

    V, Q = compute_values(traj_set, None, None, eval_qnet, config, max_length=config.max_length, model_type='IS')
    ips = importance_sampling(traj_set)
    wis = importance_sampling(traj_set, wis=True)
    sis = importance_sampling(traj_set, wis=False, soften=True)
    swis = importance_sampling(traj_set, wis=True, soften=True)
    pdis = doubly_robust(traj_set, V, Q, config, wis=False, soften=False)
    wpdis = doubly_robust(traj_set, V, Q, config, wis=True, soften=False)
    spdis = doubly_robust(traj_set, V, Q, config, wis=False, soften=True)
    swpdis = doubly_robust(traj_set, V, Q, config, wis=True, soften=True)

    time_now = time.time()
    time_dr = time_now - time_pre
    time_pre = time_now

    # eval_qnetをonlineで動かして報酬性能を得る（oracle）
    for i_episode in range(config.sample_num_traj_eval):
        values = deque()
        for i_trials in range(1):
            env.seed(seedvec[i_episode].item())
            state = preprocess_state(np.append(env.reset(), np.random.normal(size = noise_dim)), state_dim)
            true_state = state
            true_done = False
            true_steps = 0
            true_rewards = 0
            while not true_done:
                true_action = epsilon_greedy_action(true_state, eval_qnet, 0, config.action_size)
                true_next_state, true_reward, true_done, _ = env.step(true_action.item())
                true_state = preprocess_state(np.append(true_next_state, np.random.normal(size = noise_dim)), state_dim)
                true_steps += 1
                true_rewards += true_reward
            values.append(true_rewards)
        target[i_episode] = np.mean(values)
    time_now = time.time()
    time_gt = time_now - time_pre

    print('| Sampling: {:.3f}s | Preprocess for MRDR: {:.3f}s | Learn MDP: {:.3f}s | Learn TC: {:.3f}s '
          '| Eval MDP: {:.3f}s | Eval DR: {:.3f}s | Eval: {:.3f}s |'
          .format(time_sampling, time_premrdr, time_mdp, time_tc, time_eval, time_dr, time_gt))

    # print('| Sampling: {:.3f}s | Preprocess for MRDR: {:.3f}s | Learn MRDR: {:.3f}s | Learn MDP: {:.3f}s | Learn TC: {:.3f}s '
    #       '| Eval MDP: {:.3f}s | Eval DR: {:.3f}s | Eval: {:.3f}s |'
    #       .format(time_sampling, time_premrdr, time_mrdr, time_mdp, time_tc, time_eval, time_dr, time_gt))
    print('Target policy value:', np.mean(target))

    results = [dml_dr_cross_k_nd,
               dml_dr_cross_k_estpz_nd,
               dml_dr_cross_k_estpz_wis_nd,
               dml_dr_cross_k_estpz_sis_nd,
               dml_dr_cross_k_estpz_swis_nd,
               dml_dr_cross_k_chunk_nd,
               mv_bsl,
               dr_bsl,
               dr_bsl_estpz,
               wdr_bsl_estpz,
               wdr_bsl,
               sdr_bsl,
               swdr_bsl,
               ips,
               wis,
               sis,
               swis,
               pdis,
               wpdis,
               spdis,
               swpdis]
    return results, target
