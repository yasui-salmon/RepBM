import math
import random
from collections import deque
import pandas as pd

import gym
import numpy as np
import torch
import torch.optim as optim

from src.config import acrobot_config
from src.models import QNet
from src.utils import save_qnet, select_maxq_action
from src.memory import ReplayMemory, Transition

import warnings
warnings.filterwarnings('ignore')

# if gpu is to be used
use_cuda = False #torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


def epsilon_decay_per_ep(step, config):
    return config.dqn_epsilon_min + (config.dqn_epsilon - config.dqn_epsilon_min)*math.exp(-0.001*step)


def preprocess_state(state, state_dim):
    return Tensor(np.reshape(state, [1, state_dim]))


def select_action(state, qnet, epsilon, action_size):
    sample = random.random()
    if sample > epsilon:
        # return qnet(
        #     state.type(FloatTensor)).detach().max(1)[1].view(1,1)

        return select_maxq_action(state, qnet)
    else:
        return LongTensor([[random.randrange(action_size)]])


def guided_action(state, qnet, epsilon, action_size):
    if random.random() < 0.1:
        return LongTensor([[random.randrange(action_size)]])
    if state[0,1]>0:
        return LongTensor([[2]])
    else:
        return LongTensor([[0]])


def replay_and_optim(memory, qnet, target_net, optimizer, criterion, config):
    # Adapted from pytorch tutorial example code
    if config.dqn_batch_size > len(memory):
        return 0
    transitions = memory.sample(config.dqn_batch_size)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = ByteTensor(tuple(map(lambda d: d is False, batch.done)))
    non_final_next_states = torch.cat([t.next_state for t in transitions if t.done is False])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = qnet(state_batch.type(Tensor)).gather(1, action_batch).squeeze()

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(config.dqn_batch_size).type(Tensor)
    next_state_values[non_final_mask] = qnet(non_final_next_states.type(Tensor)).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * config.gamma) + reward_batch
    # Undo volatility (which was used to prevent unnecessary gradients)
    expected_state_action_values = expected_state_action_values.detach()

    # Compute Huber loss
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in qnet.parameters():
       param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss.item()


if __name__ == "__main__":
    env = gym.make("Acrobot-v1")
    config = acrobot_config

    print(config.state_dim,config.dqn_hidden_dims,config.action_size)

    qnet = QNet(config.state_dim,config.dqn_hidden_dims,config.action_size)
    target_net = QNet(config.state_dim,config.dqn_hidden_dims,config.action_size)
    memory = ReplayMemory(config.buffer_capacity)
    epsilon = config.dqn_epsilon
    scores = deque(maxlen=100)
    dp_scores = deque(maxlen=100)
    reward_score = deque(maxlen=100)
    optimizer = optim.Adam(qnet.parameters(),lr=config.dqn_alpha)

    learning_result = []

    criterion = torch.nn.MSELoss()
    for i_episode in range(config.dqn_num_episodes):
        # Initialize the environment and state
        state = preprocess_state(env.reset(),config.state_dim)
        done = False
        n_steps = 0

        while not done:
            # Select and perform an action
            action = select_action(state, qnet, epsilon=epsilon, action_size=config.action_size)
            # observe next state and reward
            next_state, reward, done, _ = env.step(action.item())

            #print(next_state.shape, reward, done)
            next_state = preprocess_state(next_state,config.state_dim)
            reward = Tensor([reward])

            if done:
                reward = Tensor([499-n_steps])
            else:
                reward = Tensor([0])

            # Store the transition in memory
            memory.push(state, action, next_state, reward, done, None)
            # Move to the next state
            state = next_state
            n_steps += 1
        scores.append(n_steps)

        state = preprocess_state(env.reset(), config.state_dim)
        done = False
        n_steps = 0

        while not done:
            # Select and perform an action
            action = select_maxq_action(state, qnet)
            next_state, reward, done, _ = env.step(action.item())

            if done:
                reward = Tensor([499-n_steps])
            else:
                reward = Tensor([0])

            next_state = preprocess_state(next_state, config.state_dim)
            state = next_state
            n_steps += 1
        dp_scores.append(n_steps)
        reward_score.append(reward)
        mean_score = np.mean(dp_scores)
        mean_reward = np.mean(reward_score)

        loss = replay_and_optim(memory, qnet, target_net, optimizer, criterion, config)

        if mean_reward > 420 and i_episode >= 100:
            print('Ran {} episodes. Solved after {} trials ✔'.format(i_episode, i_episode - 100))
            break

        if i_episode % 100 == 0:
            print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks. Epsilon is {:.3e}. Loss is {:.3e}. reward is {}'
                  .format(i_episode, mean_score, epsilon, loss, mean_reward))

            out = {"i_episode":i_episode, "mean_score": mean_score, "epsilon":epsilon, "loss":loss, "mean_reward":mean_reward}
            learning_result.append(out)
            pd.DataFrame(learning_result).to_csv("qlearning_acrobat_lr.csv")
        # update
        epsilon = epsilon_decay_per_ep(i_episode,config)
    save_qnet(state={'state_dict': qnet.state_dict()})

    groundtruth = deque()
    for i_episode in range(1000):
        true_state = preprocess_state(env.reset(), config.state_dim)
        true_done = False
        true_steps = 0
        while not true_done:
            # action = select_action_random(action_size=config.action_size)
            action = select_maxq_action(true_state, qnet)
            true_next_state, true_reward, true_done, _ = env.step(action.item())
            true_steps += 1
            true_state = preprocess_state(true_next_state, config.state_dim)
        groundtruth.append(true_steps)
    print('True survival time is {} with std dev {:.3e}'.format(np.mean(groundtruth),
                                                                np.std(groundtruth) / len(groundtruth)))
