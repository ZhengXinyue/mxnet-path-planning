#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 环境模型：env_0_obstacle .py
# launch文件：one_jackal_image_add_sensor.launch
# world文件：empty_sensor.world


import os
import datetime
import time
import random
import math
from collections import deque


import numpy as np
import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import autograd, nd, init, gluon
from mxnet.gluon import loss as gloss, nn, rnn, data as gdata
import gluonbook as gb

from my_env import envmodel


env = envmodel()
ctx = mx.gpu()
seed = 1
random.seed(seed)
np.random.seed(seed)
mx.random.seed(seed)


# (v, w)
action_dict = {0: [1.0, -1.0], 1: [1.0, -0.5], 2: [1.0, 0.0],
               3: [1.0, 0.5], 4: [1.0, 1.0], 5: [0.5, -1.0],
               6: [0.5, 0.0], 7: [0.5, 1.0], 8: [0.0, -1.0],
               9: [0.0, 0.0], 10: [0.0, 1.0]}


class Dueling_network(gluon.nn.Block):
    def __init__(self, n_actions, **kwargs):
        super(Dueling_network, self).__init__(**kwargs)
        self.n_actions = n_actions
        self.conv0 = nn.Conv2D(32, kernel_size=8, strides=4, padding=2, activation='relu')
        self.conv1 = nn.Conv2D(64, kernel_size=4, strides=2, padding=1, activation='relu')
        self.conv2 = nn.Conv2D(64, kernel_size=3, strides=1, padding=1, activation='relu')
        self.a_dense0 = nn.Dense(512, activation='relu')
        self.a_dense1 = nn.Dense(self.n_actions)
        self.v_dense0 = nn.Dense(512, activation='relu')
        self.v_dense1 = nn.Dense(1)

    def forward(self, visual, self_state):
        feature = nd.flatten(self.conv2(self.conv1(self.conv0(visual))))
        # 2896 + 6400
        input = nd.concat(feature, self_state, dim=1)
        # batch_size x n_actions
        a_value = self.a_dense1(self.a_dense0(input))
        # batch_size x 1
        v_value = self.v_dense1(self.v_dense0(input))
        # batch_size x 1
        mean_value = (nd.sum(a_value, axis=1) / self.n_actions).reshape((visual.shape[0], 1))
        # broadcast
        q_value = v_value + a_value - mean_value
        return q_value


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # sample one leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s < self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1

        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priorityinfo = input()
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[dataIdx]


class MemoryBuffer:
    e = 0.00001
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, size, ctx):
        self.tree = SumTree(size)
        self.size = size
        self.buffer = deque(maxlen=self.size)
        self.ctx = ctx

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.alpha

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def __getitem__(self, item):
        return self.buffer[item]

    def store_transition(self, error, transition):
        p = self._get_priority(error)
        self.tree.add(p, transition)

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        # batch size x 4 x 80 x 80
        visual_state_batch = nd.array([data[0] for data in batch], ctx=self.ctx)
        # batch size x 16
        lidar_state_batch = nd.array([data[1] for data in batch], ctx=self.ctx).flatten()
        # 1 x batch size
        action_batch = nd.array([data[2] for data in batch], ctx=self.ctx)
        reward_batch = nd.array([data[3] for data in batch], ctx=self.ctx)

        next_visual_state_batch = nd.array([data[4] for data in batch], ctx=self.ctx)
        next_lidar_state_batch = nd.array([data[5] for data in batch], ctx=self.ctx).flatten()

        return visual_state_batch, lidar_state_batch, action_batch, reward_batch, \
               next_visual_state_batch, next_lidar_state_batch, idxs, is_weight


class D3QN_PER:
    def __init__(self,
                 n_actions,
                 explore_steps,
                 clip_theta,
                 learning_rate,
                 init_epsilon,
                 final_epsilon,
                 gamma,
                 buffer_size,
                 batch_size,
                 replace_iter,
                 annealing_end,
                 tau,
                 ctx):
        self.n_actions = n_actions
        self.explore_stpes = explore_steps
        self.clip_theta = clip_theta
        self.learning_rate = learning_rate
        self.init_epsilon = init_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon = init_epsilon
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.replace_iter = replace_iter
        self.annealing_end = annealing_end
        self.tau = tau    # for soft update
        self.ctx = ctx
        self.loss = []
        self.max_q_value_list = []
        self.load = 0   # 0 if not loaded,    1 if loaded
        self.episode = 0

        self.total_steps = 0
        self.replay_buffer = MemoryBuffer(buffer_size, self.ctx)

        self.target_network = Dueling_network(self.n_actions)
        self.main_network = Dueling_network(self.n_actions)

        self.target_network.collect_params().initialize(init=init.Xavier(), ctx=self.ctx)
        self.main_network.collect_params().initialize(init=init.Xavier(), ctx=self.ctx)

        self.optimizer = gluon.Trainer(self.main_network.collect_params(), 'sgd', {'learning_rate': self.learning_rate})

    def choose_action(self, visual_state, lidar_state):
        visual_state = nd.array([visual_state], ctx=self.ctx)
        # 1 x 16
        lidar_state = nd.array([lidar_state], ctx=self.ctx).flatten()
        if self.total_steps < self.explore_stpes:
            action = random.choice(range(self.n_actions))
        else:
            if nd.random.uniform(0, 1) > self.epsilon:
                q_value = self.main_network(visual_state, lidar_state)
                print(q_value)
                max_q_value_action = nd.argmax(q_value, axis=1)
                max_q_value = nd.max(q_value, axis=1)
                self.max_q_value_list.append(float(max_q_value.asnumpy()))
                action = int(max_q_value_action.asnumpy())
            else:
                action = random.choice(range(self.n_actions))
            # anneal
            self.epsilon = max(self.final_epsilon, self.epsilon - (self.init_epsilon - self.final_epsilon) / self.annealing_end)
        self.total_steps += 1
        return action

    def update_params(self):
        visual_state_batch, lidar_state_batch, action_batch, reward_batch, \
        next_visual_state_batch, next_lidar_state_batch, idxs, is_weight = self.replay_buffer.sample(self.batch_size)
        with autograd.record():
            # main network Q(s,a)
            current_state_q_value = self.main_network(visual_state_batch, lidar_state_batch)
            main_q_value = nd.pick(current_state_q_value, action_batch)

            # target network Q(s,a)
            next_state_q_value = self.target_network(next_visual_state_batch, next_lidar_state_batch).detach()
            max_action_batch = nd.argmax(current_state_q_value, axis=1)
            target_q_value = nd.pick(next_state_q_value, max_action_batch).detach()
            target_q_value = reward_batch + self.gamma * target_q_value

            errors = nd.abs(target_q_value - main_q_value).asnumpy()
            for i in range(self.batch_size):
                idx = idxs[i]
                self.replay_buffer.update(idx, errors[i])

            loss = gloss.L2Loss()
            l = loss(target_q_value, main_q_value) * nd.array([is_weight], ctx=self.ctx).mean()
            self.loss.append(float(l.mean().asnumpy()))
        l.backward()
        params = [p.data() for p in self.main_network.collect_params().values()]
        gb.grad_clipping(params, theta=self.clip_theta, ctx=self.ctx)
        self.optimizer.step(1)

    def hard_replace(self):
        self.main_network.save_parameters('temp_params')
        self.target_network.load_parameters('temp_params')
        print('parameters hard replaced')

    def soft_replace(self):
        value1 = self.target_network.collect_params().keys()
        value2 = self.main_network.collect_params().keys()
        d = zip(value1, value2)
        for x, y in d:
            self.target_network.collect_params()[x].data()[:] = \
                self.target_network.collect_params()[x].data() * (1 - self.tau) + \
                self.main_network.collect_params()[y].data() * self.tau

    def save_model(self):
        self.target_network.save_parameters('%s/target network parameters at episode %d' % (time, self.episode))
        self.main_network.save_parameters('%s/main network parameters at episode %d' % (time, self.episode))

    def save_final_model(self):
        self.target_network.save_parameters('%s/final target network parameters' % time)
        self.main_network.save_parameters('%s/final main network parameters' % time)

    def load_model(self):
        self.load = 1
        self.main_network.load_parameters(load_model_path1)
        self.target_network.load_parameters(load_model_path2)


episode = 0
episode_reward_list = []
cmd = [0.0, 0.0]    # command for navigate (v, w)
initialized = False
success_times = 0
agent = D3QN_PER(n_actions=len(action_dict),
                 explore_steps=0,
                 clip_theta=10,
                 learning_rate=0.0005,
                 init_epsilon=1,
                 final_epsilon=0.05,
                 gamma=0.99,
                 buffer_size=5000,
                 batch_size=32,
                 replace_iter=500,
                 annealing_end=10000,
                 tau=0.001,
                 ctx=ctx)


def get_initial_coordinate():
    while True:
        start_end_point = 2 * d * np.random.random_sample((2, 2)) - d     # (-1, 1) * d
        if math.sqrt((start_end_point[0][0] - start_end_point[1][0]) ** 2 +
                     (start_end_point[0][1] - start_end_point[1][1]) ** 2) > 10:
            break
    return start_end_point


time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
os.mkdir(time)
load_model_path1 = '2019-10-31 16:18:34/final main network parameters'
load_model_path2 = '2019-10-31 16:18:34/final target network parameters'
# agent.load_model()

d = 10     # the distance from start point to goal point
max_episode_steps = 200
max_episodes = 500

for episode in range(max_episodes):
    agent.episode = episode
    if episode > 0 and episode % 50 == 0:
        agent.save_model()
    episode_steps = 0
    episode_reward = 0

    start_end_point = get_initial_coordinate()
    env.reset_env(start=[start_end_point[0][0], start_end_point[0][1]],
                  goal=[0, 0])
    # env.reset_env(start=[start_end_point[0][0], start_end_point[0][1]],
                  # goal=[start_end_point[1][0], start_end_point[1][1]])
    env_info = env.get_env()

    # use deque to stack
    state_deque = deque(maxlen=4)
    visual_deque = deque(maxlen=4)

    # lidar: 724  list
    # visual: 80x80  np.array
    lidar, visual = np.array(env_info[0]), (env_info[1][np.newaxis, :] - (255 / 2)) / (255 / 2)
    # self state 1 x 724
    self_state = lidar[:][np.newaxis, :]
    terminal, reward = env_info[2], env_info[3]
    # initialize the first state
    for i in range(4):
        visual_deque.append(visual)
    # 4x3x3  np.array
    visual_state = np.concatenate((visual_deque[0], visual_deque[1], visual_deque[2], visual_deque[3]), axis=0)
    for i in range(4):
        state_deque.append(self_state)
    # 4x724  np.array
    all_self_state = np.concatenate((state_deque[0], state_deque[1], state_deque[2], state_deque[3]), axis=0)
    # 1 x 2896
    all_self_state = all_self_state.reshape(1, -1)

    episode_steps += 1

    for step in range(max_episode_steps):
        action = agent.choose_action(visual_state, all_self_state)
        # initialize the network parameters with one forward
        if not initialized:
            m = nd.array([visual_state], ctx=ctx)
            n = nd.array([all_self_state], ctx=ctx).flatten()
            agent.target_network(m, n)
            agent.main_network(m, n)
            initialized = True
        v_cmd = action_dict[action][0]
        w_cmd = action_dict[action][1]
        cmd[0] = v_cmd
        cmd[1] = w_cmd
        env.step(cmd)

        env_info = env.get_env()
        lidar, visual, terminal, reward = env_info[0], env_info[1], env_info[2], env_info[3]
        lidar = np.array(lidar)
        self_state = lidar[:][np.newaxis, :]
        visual = (np.array(visual)[np.newaxis, :] - (255 / 2)) / (255 / 2)
        state_deque.append(self_state)
        visual_deque.append(visual)

        next_visual_state = np.concatenate((visual_deque[0], visual_deque[1], visual_deque[2], visual_deque[3]), axis=0)
        next_all_self_state = np.concatenate((state_deque[0], state_deque[1], state_deque[2], state_deque[3]), axis=0)
        next_all_self_state = next_all_self_state.reshape(1, -1)

        transition = (visual_state, all_self_state, action, reward, next_visual_state, next_all_self_state)

        # compute the TD_error
        m = nd.array([visual_state], ctx=ctx)
        j = nd.array([all_self_state], ctx=ctx).flatten()
        main_q_value = agent.main_network(m, j)
        chosen_main_q_value = nd.pick(main_q_value, nd.array([action], ctx=ctx))
        target_q_value = agent.target_network(m, j)
        target_action = nd.argmax(target_q_value, axis=1)
        chosen_target_q_value = nd.pick(main_q_value, target_action)

        m = chosen_main_q_value - reward - agent.gamma * chosen_target_q_value
        error = m.asnumpy()

        agent.replay_buffer.store_transition(error, transition)

        if agent.total_steps > agent.explore_stpes:    # attention
            agent.update_params()
            if agent.total_steps % agent.replace_iter == 0:
                agent.hard_replace()

        episode_reward += reward
        if episode_reward > 1:         # attention
            success_times += 1
        episode_steps += 1

        if episode_steps == max_episode_steps:
            terminal = True
        if terminal:
            break
        visual_state, all_self_state = next_visual_state, next_all_self_state

    print('episode: ' + str(episode) + ' reward: ' + str(episode_reward) +
          ' episode steps: ' + str(episode_steps) + ' epsilon: ' + str(agent.epsilon))
    episode_reward_list.append(episode_reward)
agent.save_final_model()

# log information
# random seed, d, total episodes, max episode steps,
# init epsilon, final epsilon, gamma, buffer size, batch size, replace iter, annealing end,
# loss list, episode reward list, success times


def log_information():
    with open('%s/test.txt' % time, 'a+') as f:
        f.write('if load model:   %d\n'
                'model path:    %s\n'
                'random seed:   %d\n'
                'distance:   %d\n'
                'total episodes:   %d\n'
                'max episode steps:   %d\n'
                'explore steps:   %d\n'
                'init epsilon:   %.2f\n'
                'final epsilon:   %.2f\n'
                'gamma:   %.2f\n'
                'buffer size:   %d\n'
                'batch size:   %d\n'
                'replace iter:   %d\n'
                'annealing end:   %d\n'
                'success times:   %d\n'
                'learning rate:   %.5f\n'
                'gradient clip theta:   %.2f\n' %
                (agent.load, load_model_path1, seed, d, max_episodes, max_episode_steps, agent.explore_stpes, agent.init_epsilon, agent.final_epsilon,
                 agent.gamma, agent.buffer_size, agent.batch_size, agent.replace_iter, agent.annealing_end,
                 success_times, agent.learning_rate, agent.clip_theta))

    with open('%s/test.txt' % time, 'a+') as f:
        f.write('loss:\n' + str(agent.loss) + '\n')
        f.write('reward list:\n' + str(episode_reward_list) + '\n')
        f.write('q value: \n' + str(agent.max_q_value_list) + '\n')


log_information()

plt.plot(episode_reward_list)
plt.xlabel('episode')
plt.ylabel('reward')
plt.savefig('%s/reward' % time)
