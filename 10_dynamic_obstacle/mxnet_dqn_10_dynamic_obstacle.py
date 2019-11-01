# coding=utf-8


# 环境模型：10_dynamic_obstacle.py
# launch文件：ten_jackal_laser_add_apriltag.launch
# world文件：10obstacle_sensor_30m_5jackal.world
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
from mxnet.gluon import loss as gloss, nn, rnn
import gluonbook as gb

from env_10_dynamic_obstacle import envmodel

env = envmodel()
ctx = mx.gpu()
random.seed(1)
np.random.seed(1)
mx.random.seed(1)


# (v, w)
action_dict = {0: [1.0, -1.0], 1: [1.0, -0.5], 2: [1.0, 0.0],
               3: [1.0, 0.5], 4: [1.0, 1.0], 5: [0.5, -1.0],
               6: [0.5, 0.0], 7: [0.5, 1.0], 8: [0.0, -1.0],
               9: [0.0, 0.0], 10: [0.0, 1.0]}

d = 3     # the distance from start point to goal point


# get start coordinate
def get_initial_coordinate():
    while True:
        start_end_point = 2 * d * np.random.random_sample((2, 2)) - d
        if math.sqrt((start_end_point[0][0] - start_end_point[1][0]) ** 2 +
                     (start_end_point[0][1] - start_end_point[1][1]) ** 2) > 3:
            break
    return start_end_point


class Double_network(gluon.nn.Block):
    def __init__(self, n_actions, **kwargs):
        super(Double_network, self).__init__(**kwargs)
        self.n_actions = n_actions
        self.conv0 = nn.Conv2D(32, kernel_size=8, strides=4, padding=2)
        self.conv1 = nn.Conv2D(64, kernel_size=4, strides=2, padding=1)
        self.conv2 = nn.Conv2D(64, kernel_size=3, strides=1, padding=1)
        # self.lstm_layer = rnn.LSTM(512)
        self.dense0 = nn.Dense(512, activation='relu')
        self.dense1 = nn.Dense(256, activation='relu')
        self.dense2 = nn.Dense(self.n_actions)

    def forward(self, visual, lidar):
        x = nd.flatten(self.conv2(self.conv1(self.conv0(visual))))
        # y = nd.reshape(self.lstm_layer(lidar), (-1, 2048))    # attention
        # z = nd.concat(x, y, dim=1)
        q_value = self.dense2(self.dense1(self.dense0(x)))
        return q_value


class MemoryBuffer:
    def __init__(self, size, ctx):
        self.size = size
        self.buffer = deque(maxlen=self.size)
        self.ctx = ctx

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def __getitem__(self, item):
        return self.buffer[item]

    def store_transition(self, visual_state, lidar_state, action, reward, next_visual_state, next_lidar_state):
        transition = (visual_state, lidar_state, action, reward, next_visual_state, next_lidar_state)
        self.buffer.append(transition)

    def sample(self, batch_size):
        assert len(self.buffer) > batch_size
        minibatch = random.sample(self.buffer, batch_size)
        # batch size x 4 x 80 x 80
        visual_state_batch = nd.array([data[0] for data in minibatch], ctx=self.ctx)
        # batch size x 4 x 724
        lidar_state_batch = nd.array([data[1] for data in minibatch], ctx=self.ctx)
        # change to 4 x batch size x 724
        lidar_state_batch = nd.swapaxes(lidar_state_batch, 0, 1)
        # 1 x batch size
        action_batch = nd.array([data[2] for data in minibatch], ctx=self.ctx)
        reward_batch = nd.array([data[3] for data in minibatch], ctx=self.ctx)

        next_visual_state_batch = nd.array([data[4] for data in minibatch], ctx=self.ctx)
        next_lidar_state_batch = nd.swapaxes(nd.array([data[5] for data in minibatch], ctx=self.ctx), 0, 1)

        return visual_state_batch, lidar_state_batch, \
               action_batch, reward_batch, next_visual_state_batch, next_lidar_state_batch


class Double_DQN:
    def __init__(self,
                 n_actions,
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
        self. init_epsilon = init_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon = init_epsilon
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.replace_iter = replace_iter
        self.annealing_end = annealing_end
        self.tau = tau    # for soft update
        self.ctx = ctx

        self.total_steps = 0
        self.replay_buffer = MemoryBuffer(buffer_size, self.ctx)

        self.target_network = Double_network(self.n_actions)
        self.main_network = Double_network(self.n_actions)

        self.target_network.collect_params().initialize(init=init.Xavier(), ctx=self.ctx)
        self.main_network.collect_params().initialize(init=init.Xavier(), ctx=self.ctx)

        self.optimizer = gluon.Trainer(self.main_network.collect_params(), 'adam')

    def choose_action(self, visual_state, lidar_state):
        visual_state = nd.array([visual_state], ctx=self.ctx)
        # change to 4 x 1 x 724
        lidar_state = nd.swapaxes(nd.array([lidar_state], ctx=self.ctx), 0, 1)
        if self.total_steps < 1000:
            action = random.choice(range(self.n_actions))
        else:
            if nd.random.uniform(0, 1) > self.epsilon:
                q_value = self.main_network(visual_state, lidar_state)
                action = int(nd.argmax(q_value, axis=1).asnumpy())
            else:
                action = random.choice(range(self.n_actions))
            # anneal
            self.epsilon = max(self.final_epsilon, self.epsilon - (self.init_epsilon - self.final_epsilon) / self.annealing_end)
        self.total_steps += 1
        return action

    def update_params(self):
        visual_state_batch, lidar_state_batch, action_batch, reward_batch, \
        next_visual_state_batch, next_lidar_state_batch = self.replay_buffer.sample(self.batch_size)
        with autograd.record():
            # main network Q(s,a)
            current_state_q_value = self.main_network(visual_state_batch, lidar_state_batch)
            main_q_value = nd.pick(current_state_q_value, action_batch)

            # target network Q(s,a)
            next_state_q_value = self.target_network(next_visual_state_batch, next_lidar_state_batch).detach()
            max_action_batch = nd.argmax(current_state_q_value, axis=1).detach()
            target_q_value = nd.pick(next_state_q_value, max_action_batch)
            target_q_value = target_q_value + self.gamma * reward_batch

            loss = gloss.L2Loss()
            l = loss(target_q_value, main_q_value)
        l.backward()
        self.optimizer.step(batch_size=self.batch_size)

    def hard_replace(self):
        self.main_network.save_parameters('DDQN_params')
        self.target_network.load_parameters('DDQN_params')
        print('Double DQN parameters hard replaced')

    def soft_replace(self):
        value1 = self.target_network.collect_params().keys()
        value2 = self.main_network.collect_params().keys()
        d = zip(value1, value2)
        for x, y in d:
            self.target_network.collect_params()[x].data()[:] = \
                self.target_network.collect_params()[x].data() * (1 - self.tau) + \
                self.main_network.collect_params()[y].data() * self.tau

    def save_model(self):
        self.target_network.save_parameters('Double DQN target network buffer size: %d '
                                            'batch size: %d '
                                            'replace iter: %d '
                                            'at steps %d' % (self.buffer_size, self.batch_size,
                                                             self.replace_iter, self.total_steps))

        self.main_network.save_parameters('Double DQN main network buffer size: %d '
                                          'batch size: %d '
                                          'replace iter: %d '
                                          'at steps %d' % (
                                           self.buffer_size, self.batch_size, self.replace_iter, self.total_steps))

    def load_model(self):
        pass


max_episode_steps = 100
max_episodes = 300
episode = 0
episode_reward_list = []
cmd = [0.0, 0.0]    # command for navigate (v, w)
already_saved = False
initialized = False
agent = Double_DQN(n_actions=len(action_dict),
                   init_epsilon=1,
                   final_epsilon=0.1,
                   gamma=0.99,
                   buffer_size=3000,
                   batch_size=32,
                   replace_iter=1000,
                   annealing_end=10000,
                   tau=0.001,
                   ctx=ctx)


for episode in range(max_episodes):
    episode_steps = 0
    episode_reward = 0

    start_end_point = get_initial_coordinate()
    env.reset_env(start=[start_end_point[0][0], start_end_point[0][1]],
                  goal=[start_end_point[1][0], start_end_point[1][1]])

    env_info, jackal_x_temp, jackal_y_temp, jackal0_x_temp, jackal0_y_temp, jackal1_x_temp, jackal1_y_temp, \
        jackal2_x_temp, jackal2_y_temp, jackal3_x_temp, jackal3_y_temp, \
        jackal4_x_temp, jackal4_y_temp, jackal5_x_temp, jackal5_y_temp, jackal6_x_temp, jackal6_y_temp, \
        jackal7_x_temp, jackal7_y_temp, jackal8_x_temp, jackal8_y_temp, jackal9_x_temp, jackal9_y_temp = env.get_env()

    # use deque to stack
    lidar_deque = deque(maxlen=4)
    visual_deque = deque(maxlen=4)

    # lidar: 724  list
    # visual: 80x80  np.array
    lidar, visual = np.array(env_info[0])[np.newaxis, :], env_info[1][np.newaxis, :]
    terminal, reward = env_info[2], env_info[3]

    # initialize the first state
    for i in range(4):
        visual_deque.append(visual)
    # 4x3x3  np.array
    visual_state = np.concatenate((visual_deque[0], visual_deque[1], visual_deque[2], visual_deque[3]), axis=0)
    for i in range(4):
        lidar_deque.append(lidar)
    # 4x724  np.array
    lidar_state = np.concatenate((lidar_deque[0], lidar_deque[1], lidar_deque[2], lidar_deque[3]), axis=0)

    episode_step = 1

    for step in range(max_episode_steps):
        action = agent.choose_action(visual_state, lidar_state)
        if not initialized:
            m = nd.array([visual_state], ctx=ctx)
            # change to 4 x 1 x 724
            n = nd.swapaxes(nd.array([lidar_state], ctx=ctx), 0, 1)
            agent.target_network(m, n)
            agent.main_network(m, n)
            initialized = True
        v_cmd = action_dict[action][0]
        w_cmd = action_dict[action][1]
        cmd[0] = v_cmd
        cmd[1] = w_cmd
        env.step(cmd)
        env.run()

        env_info, jackal_x_temp, jackal_y_temp, jackal0_x_temp, jackal0_y_temp, jackal1_x_temp, jackal1_y_temp, \
            jackal2_x_temp, jackal2_y_temp, jackal3_x_temp, jackal3_y_temp, \
            jackal4_x_temp, jackal4_y_temp, jackal5_x_temp, jackal5_y_temp, jackal6_x_temp, jackal6_y_temp, \
            jackal7_x_temp, jackal7_y_temp, jackal8_x_temp, jackal8_y_temp, \
            jackal9_x_temp, jackal9_y_temp = env.get_env()
        lidar, visual, terminal, reward = env_info[0], env_info[1], env_info[2], env_info[3]
        lidar = np.array(lidar)[np.newaxis, :]
        visual = np.array(visual)[np.newaxis, :]
        lidar_deque.append(lidar)
        visual_deque.append(visual)

        next_visual_state = np.concatenate((visual_deque[0], visual_deque[1], visual_deque[2], visual_deque[3]), axis=0)
        next_lidar_state = np.concatenate((lidar_deque[0], lidar_deque[1], lidar_deque[2], lidar_deque[3]), axis=0)

        agent.replay_buffer.store_transition(visual_state, lidar_state, action, reward,
                                             next_visual_state, next_lidar_state)

        if agent.total_steps > 1000:
            agent.update_params()
            if agent.total_steps % agent.replace_iter == 0:
                agent.hard_replace()

        episode_reward += reward
        episode_step += 1

        if episode_step == max_episode_steps:
            terminal = True
        if terminal:
            break
        visual_state, lidar_state = next_visual_state, next_lidar_state

    print('episode: ' + str(episode) + ' reward: ' + str(episode_reward) +
          ' episode steps: ' + str(episode_step) + ' epsilon: ' + str(agent.epsilon))
    if len(episode_reward_list) > 10:
        m = episode_reward_list[-10:]
        if sum(m) > 180 and not already_saved:
            agent.save_model()
            already_saved = True
    episode_reward_list.append(episode_reward)

print(episode_reward_list)
plt.xlabel('episode')
plt.ylabel('reward')
plt.title('Double DQN')
plt.savefig('./test.jpg')
plt.show()


