#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 环境模型：env_0_obstacle .py
# launch文件：one_jackal_image_add_sensor.launch
# world文件：obstacle_sensor.world


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


class MemoryBuffer:
    def __init__(self, buffer_size, ctx):
        self.buffer = deque(maxlen=buffer_size)
        self.maxsize = buffer_size
        self.ctx = ctx

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def sample(self, batch_size):
        assert len(self.buffer) > batch_size
        minibatch = random.sample(self.buffer, batch_size)
        # batch size x 4 x 80 x 80
        visual_state_batch = nd.array([data[0] for data in minibatch], ctx=self.ctx)
        # batch size x 2896
        lidar_self_state_batch = nd.array([data[1] for data in minibatch], ctx=self.ctx).flatten()
        # batch size x 2
        action_batch = nd.array([data[2] for data in minibatch], ctx=self.ctx)
        # batch size
        reward_batch = nd.array([data[3] for data in minibatch], ctx=self.ctx)
        # batch size x 4 x 80 x 80
        next_visual_state_batch = nd.array([data[4] for data in minibatch], ctx=self.ctx)
        # batch size x 2896
        next_lidar_self_state_batch = nd.array([data[5] for data in minibatch], ctx=self.ctx).flatten()
        # batch size
        done_batch = nd.array([data[6] for data in minibatch], ctx=self.ctx)

        return visual_state_batch, lidar_self_state_batch, action_batch, reward_batch, \
               next_visual_state_batch, next_lidar_self_state_batch, done_batch

    def store_transition(self, transition):
        self.buffer.append(transition)


class Actor(nn.Block):
    def __init__(self, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_dim = action_dim
        self.action_bound = action_bound

        self.conv0 = nn.Conv2D(32, kernel_size=8, strides=4, padding=2, activation='relu')
        self.conv1 = nn.Conv2D(64, kernel_size=4, strides=2, padding=1, activation='relu')
        self.conv2 = nn.Conv2D(64, kernel_size=3, strides=1, padding=1, activation='relu')
        self.dense0 = nn.Dense(512, activation='relu')
        self.dense1 = nn.Dense(128, activation='relu')
        self.v_dense = nn.Dense(1, activation='sigmoid')
        self.w_dense = nn.Dense(1, activation='tanh')

    def forward(self, visual, lidar):
        visual = visual / 255
        # batch size x 6400
        visual_feature = self.conv2(self.conv1(self.conv0(visual))).flatten()
        # batch size x (6400 + 724 x 4)
        dense_input = nd.concat(visual_feature, lidar, dim=1)
        m = self.dense1(self.dense0(dense_input))
        v_action = self.v_dense(m)
        w_action = self.w_dense(m)
        action = nd.concat(v_action, w_action, dim=1)
        upper_bound = self.action_bound[:, 1]
        action = action * upper_bound
        return action


class Critic(nn.Block):
    def __init__(self):
        super(Critic, self).__init__()

        self.conv0 = nn.Conv2D(32, kernel_size=8, strides=4, padding=2, activation='relu')
        self.conv1 = nn.Conv2D(64, kernel_size=4, strides=2, padding=1, activation='relu')
        self.conv2 = nn.Conv2D(64, kernel_size=3, strides=1, padding=1, activation='relu')
        self.dense0 = nn.Dense(512, activation='relu')
        self.dense1 = nn.Dense(128, activation='relu')
        self.dense2 = nn.Dense(1)

    def forward(self, visual, lidar, action):
        visual = visual / 255
        visual_feature = self.conv2(self.conv1(self.conv0(visual))).flatten()
        # batch size x (6400 + 724 x 4 + 2)
        dense_input = nd.concat(visual_feature, lidar, action, dim=1)
        q_value = self.dense2(self.dense1(self.dense0(dense_input)))
        return q_value


class TD3:
    def __init__(self,
                 action_dim,
                 action_bound,
                 actor_learning_rate,
                 critic_learning_rate,
                 batch_size,
                 memory_size,
                 gamma,
                 tau,
                 explore_steps,
                 policy_update,
                 policy_noise,
                 explore_noise,
                 noise_clip,
                 grad_clip,
                 ctx):
        self.action_dim = action_dim
        self.action_bound = nd.array(action_bound, ctx=ctx)

        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.gamma = gamma
        self.tau = tau
        self.explore_steps = explore_steps
        self.policy_update = policy_update
        self.policy_noise = policy_noise
        self.explore_noise = explore_noise
        self.noise_clip = noise_clip
        self.ctx = ctx
        self.grad_clip = grad_clip
        self.load = 0

        self.main_actor_network = Actor(action_dim, self.action_bound)
        self.target_actor_network = Actor(action_dim, self.action_bound)
        self.main_critic_network1 = Critic()
        self.target_critic_network1 = Critic()
        self.main_critic_network2 = Critic()
        self.target_critic_network2 = Critic()

        self.main_actor_network.collect_params().initialize(init=init.Xavier(), ctx=ctx)
        self.target_actor_network.collect_params().initialize(init=init.Xavier(), ctx=ctx)
        self.main_critic_network1.collect_params().initialize(init=init.Xavier(), ctx=ctx)
        self.target_critic_network1.collect_params().initialize(init=init.Xavier(), ctx=ctx)
        self.main_critic_network2.collect_params().initialize(init=init.Xavier(), ctx=ctx)
        self.target_critic_network2.collect_params().initialize(init=init.Xavier(), ctx=ctx)

        self.actor_optimizer = gluon.Trainer(self.main_actor_network.collect_params(),
                                             'adam',
                                             {'learning_rate': self.actor_learning_rate})
        self.critic1_optimizer = gluon.Trainer(self.main_critic_network1.collect_params(),
                                               'adam',
                                               {'learning_rate': self.critic_learning_rate})
        self.critic2_optimizer = gluon.Trainer(self.main_critic_network2.collect_params(),
                                               'adam',
                                               {'learning_rate': self.critic_learning_rate})

        self.total_steps = 0
        self.total_train_steps = 0
        self.episode = 0

        self.memory_buffer = MemoryBuffer(buffer_size=self.memory_size, ctx=ctx)

    def choose_action_train(self, visual, lidar):
        visual = nd.array([visual], ctx=self.ctx)
        lidar = nd.array([lidar], ctx=self.ctx).flatten()
        action = self.main_actor_network(visual, lidar)
        print(action)
        # no noise clip
        noise = nd.normal(loc=0, scale=self.explore_noise, shape=action.shape, ctx=self.ctx)
        action += noise
        clipped_action = self.action_clip(action).squeeze()
        return clipped_action

    def choose_action_evaluate(self, visual, lidar):
        visual = nd.array([visual], ctx=self.ctx)
        lidar = nd.array([lidar], ctx=self.ctx).flatten()
        action = self.main_actor_network(visual, lidar).squeeze()
        return action

    def action_clip(self, action):
        action0 = nd.clip(action[:, 0], a_min=float(self.action_bound[0][0].asnumpy()), a_max=float(self.action_bound[0][1].asnumpy()))
        action1 = nd.clip(action[:, 1], a_min=float(self.action_bound[1][0].asnumpy()), a_max=float(self.action_bound[1][1].asnumpy()))
        clipped_action = nd.concat(action0.reshape(-1, 1), action1.reshape(-1, 1))
        return clipped_action

    def soft_update(self, target_network, main_network):
        target_parameters = target_network.collect_params().keys()
        main_parameters = main_network.collect_params().keys()
        d = zip(target_parameters, main_parameters)
        for x, y in d:
            target_network.collect_params()[x].data()[:] = \
                target_network.collect_params()[x].data() * \
                (1 - self.tau) + main_network.collect_params()[y].data() * self.tau

    def update(self):
        self.total_train_steps += 1
        visual_state_batch, lidar_self_state_batch, action_batch, reward_batch, \
        next_visual_state_batch, next_lidar_self_state_batch, done_batch = self.memory_buffer.sample(self.batch_size)

        # --------------optimize the critic network--------------------
        with autograd.record():
            # choose next action according to target policy network
            next_action_batch = self.target_actor_network(next_visual_state_batch, next_lidar_self_state_batch)
            noise = nd.normal(loc=0, scale=self.policy_noise, shape=next_action_batch.shape, ctx=self.ctx)
            # with noise clip
            noise = nd.clip(noise, a_min=-self.noise_clip, a_max=self.noise_clip)
            next_action_batch = next_action_batch + noise
            clipped_action = self.action_clip(next_action_batch)

            # get target q value
            target_q_value1 = self.target_critic_network1(next_visual_state_batch, next_lidar_self_state_batch, clipped_action)
            target_q_value2 = self.target_critic_network2(next_visual_state_batch, next_lidar_self_state_batch, clipped_action)
            target_q_value = nd.minimum(target_q_value1, target_q_value2).squeeze()
            target_q_value = reward_batch + (1.0 - done) * (self.gamma * target_q_value)

            # get current q value
            current_q_value1 = self.main_critic_network1(visual_state_batch, lidar_self_state_batch, action_batch)
            current_q_value2 = self.main_critic_network2(visual_state_batch, lidar_self_state_batch, action_batch)

            loss = gloss.L2Loss()
            value_loss1 = loss(current_q_value1, target_q_value.detach())
            value_loss2 = loss(current_q_value2, target_q_value.detach())
            value_loss = value_loss1 + value_loss2
        self.main_critic_network1.collect_params().zero_grad()
        self.main_critic_network2.collect_params().zero_grad()
        value_loss.backward()
        params1 = [p.data() for p in self.main_critic_network1.collect_params().values()]
        gb.grad_clipping(params1, theta=self.grad_clip, ctx=self.ctx)
        params2 = [p.data() for p in self.main_critic_network2.collect_params().values()]
        gb.grad_clipping(params2, theta=self.grad_clip, ctx=self.ctx)
        self.critic1_optimizer.step(self.batch_size)
        self.critic2_optimizer.step(self.batch_size)

        # ---------------optimize the actor network-------------------------
        if self.total_train_steps % self.policy_update == 0:
            with autograd.record():
                pred_action_batch = self.main_actor_network(visual_state_batch, lidar_self_state_batch)
                actor_loss = -nd.mean(self.main_critic_network1(visual_state_batch, lidar_self_state_batch, pred_action_batch))

            self.main_actor_network.collect_params().zero_grad()
            actor_loss.backward()
            params3 = [p.data() for p in self.main_actor_network.collect_params().values()]
            gb.grad_clipping(params3, theta=self.grad_clip, ctx=self.ctx)
            self.actor_optimizer.step(1)

            self.soft_update(self.target_actor_network, self.main_actor_network)
            self.soft_update(self.target_critic_network1, self.main_critic_network1)
            self.soft_update(self.target_critic_network2, self.main_critic_network2)

    def save_model(self):
        self.main_actor_network.save_parameters(
            '%s/main actor network parameters at episode %d' % (time, self.episode))
        self.target_actor_network.save_parameters(
            '%s/target actor network parameters at episode %d' % (time, self.episode))
        self.main_critic_network1.save_parameters(
            '%s/main critic network1 parameters at episode %d' % (time, self.episode))
        self.main_critic_network2.save_parameters(
            '%s/main critic network2 parameters at episode %d' % (time, self.episode))
        self.target_critic_network1.save_parameters(
            '%s/target critic network1 parameters at episode %d' % (time, self.episode))
        self.target_critic_network2.save_parameters(
            '%s/target critic network2 parameters at episode %d' % (time, self.episode))

    def load_model(self):
        self.load = 1
        self.main_actor_network.save_parameters(
            '%s/main actor network parameters at episode %d' % (time, self.episode))
        self.target_actor_network.save_parameters(
            '%s/target actor network parameters at episode %d' % (time, self.episode))
        self.main_critic_network1.save_parameters(
            '%s/main critic network1 parameters at episode %d' % (time, self.episode))
        self.main_critic_network2.save_parameters(
            '%s/main critic network2 parameters at episode %d' % (time, self.episode))
        self.target_critic_network1.save_parameters(
            '%s/target critic network1 parameters at episode %d' % (time, self.episode))
        self.target_critic_network2.save_parameters(
            '%s/target critic network2 parameters at episode %d' % (time, self.episode))


def log_information():
    with open('%s/training log.txt' % time, 'a+') as f:
        f.write('if load model:   %d\n'
                'model path:    %s\n'
                'random seed:   %d\n'
                'distance:   %d\n'
                'total episodes:   %d\n'
                'max episode steps:   %d\n'
                'explore steps:   %d\n'
                'actor learning rate:   %.5f\n'
                'critic learning rate:    %.5f\n'
                'gamma:   %.5f\n'
                'memory size:   %d\n'
                'batch size:   %d\n'
                'tau:   %.5f\n'
                'policy noise:   %.5f\n'
                'explore noise:    %.5f\n'
                'noise clip:     %.5f\n'
                'grad clip:      %.2f\n'
                'policy update:     %d\n'
                'success times:   %d\n'
                'frame stack:   %d\n' %
                (agent.load,
                 load_model_path1,
                 seed,
                 d,
                 max_episodes,
                 max_episode_steps,
                 agent.explore_steps,
                 agent.actor_learning_rate,
                 agent.critic_learning_rate,
                 agent.gamma,
                 agent.memory_size,
                 agent.batch_size,
                 agent.tau,
                 agent.policy_noise,
                 agent.explore_noise,
                 agent.noise_clip,
                 agent.grad_clip,
                 agent.policy_update,
                 success_times,
                 n_fram_stack))

    with open('%s/training log.txt' % time, 'a+') as f:
        f.write('reward list:\n' + str(episode_reward_list) + '\n')


def get_initial_coordinate():
    while True:
        start_end_point = 2 * d * np.random.random_sample((2, 2)) - d     # (-1, 1) * d
        if math.sqrt((start_end_point[0][0] - start_end_point[1][0]) ** 2 +
                     (start_end_point[0][1] - start_end_point[1][1]) ** 2) > 5:
            break
    return start_end_point


ctx = mx.gpu()
success_times = 0
seed = 11111
random.seed(seed)
np.random.seed(seed)
mx.random.seed(seed)
episode = 0
episode_reward_list = []
time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
n_fram_stack = 4
mode = 'train'
d = 15    # the distance from start point to goal point
max_episode_steps = 300
max_episodes = 1000
target_reward = 1
agent = TD3(action_dim=2,
            action_bound=[[0, 1], [-1, 1]],
            actor_learning_rate=0.00001,
            critic_learning_rate=0.00001,
            batch_size=64,
            memory_size=100000,
            gamma=0.99,
            tau=0.005,
            explore_steps=10000,
            policy_update=2,
            policy_noise=0.2,
            explore_noise=0.3,
            noise_clip=0.3,
            grad_clip=100,
            ctx=ctx
            )


if mode == 'train':
    load_model_path1 = '2019-11-01 21:00:29/final main network parameters'
    os.mkdir(time)
    for episode in range(1, max_episodes+1):
        agent.episode += 1
        if episode % 50 == 0:
            agent.save_model()

        # use deque to stack
        state_deque = deque(maxlen=4)
        visual_deque = deque(maxlen=4)

        episode_steps = 0
        episode_reward = 0
        start_end_point = get_initial_coordinate()
        env.reset_env(start=[start_end_point[0][0], start_end_point[0][1]],
                      goal=[0, 0])
        # env.reset_env(start=[start_end_point[0][0], start_end_point[0][1]],
        # goal=[start_end_point[1][0], start_end_point[1][1]])

        env_info = env.get_env()

        # lidar_state: 724 np.array
        lidar_self = np.array(env_info[0])
        # get lidar information or self state information using slice -----------------------
        # 1x724 np.array
        lidar_self = lidar_self[:][np.newaxis, :]

        # visual: 1x80x80 np.array
        visual = env_info[1][np.newaxis, :]

        done, reward = env_info[2], env_info[3]

        # initialize the first state
        for i in range(n_fram_stack):
            visual_deque.append(visual)
        # 4x80x80  np.array
        visual_state = np.concatenate((visual_deque[0], visual_deque[1], visual_deque[2], visual_deque[3]), axis=0)

        for i in range(n_fram_stack):
            state_deque.append(lidar_self)
        # 4x724  np.array
        lidar_self_state = np.concatenate((state_deque[0], state_deque[1], state_deque[2], state_deque[3]), axis=0)
        # 1x2896 np.array
        lidar_self_state = lidar_self_state.reshape(1, -1)

        while True:
            if agent.total_steps < agent.explore_steps:
                v_cmd = random.uniform(0, 1)
                w_cmd = random.uniform(-1, 1)
                action = [v_cmd, w_cmd]
                agent.total_steps += 1
                episode_steps += 1
            else:
                action = agent.choose_action_train(visual_state, lidar_self_state)
                v_cmd = float(action[0].asnumpy())
                w_cmd = float(action[1].asnumpy())
                action = [v_cmd, w_cmd]
                agent.total_steps += 1
                episode_steps += 1

            env.step(action)
            env_info = env.get_env()
            lidar_self, visual, done, reward = env_info[0], env_info[1], env_info[2], env_info[3]
            episode_reward += reward

            # lidar information and self state information   using slice --------------------------------
            lidar_self = np.array(lidar_self)
            lidar_self = lidar_self[:][np.newaxis, :]

            visual = visual[np.newaxis, :]
            state_deque.append(lidar_self)
            visual_deque.append(visual)

            next_visual_state = np.concatenate((visual_deque[0], visual_deque[1], visual_deque[2], visual_deque[3]),
                                               axis=0)
            next_lidar_self_state = np.concatenate((state_deque[0], state_deque[1], state_deque[2], state_deque[3]),
                                                   axis=0)
            next_lidar_self_state = next_lidar_self_state.reshape(1, -1)

            transition = (visual_state, lidar_self_state,
                          action, reward,
                          next_visual_state, next_lidar_self_state,
                          done)
            agent.memory_buffer.store_transition(transition)

            visual_state, lidar_self_state = next_visual_state, next_lidar_self_state
            if agent.total_steps >= agent.explore_steps:
                agent.update()
            if done:
                break
            if episode_steps >= 300:
                break
        if episode_reward > target_reward:
            success_times += 1
        episode_reward_list.append(episode_reward)
        print('episode %d ends with reward %f' % (episode, episode_reward))
    plt.plot(episode_reward_list)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.savefig('%s/reward' % time)
    plt.show()
    log_information()


elif mode == 'test':
    load_model_path1 = '2019-11-01 21:00:29/final main network parameters'
    load_model_path2 = '2019-11-01 21:00:29/final target network parameters'
    agent.load_model()
    for episode in range(1, max_episodes + 1):
        # use deque to stack
        state_deque = deque(maxlen=4)
        visual_deque = deque(maxlen=4)

        episode_steps = 0
        episode_reward = 0
        start_end_point = get_initial_coordinate()
        env.reset_env(start=[start_end_point[0][0], start_end_point[0][1]],
                      goal=[0, 0])
        # env.reset_env(start=[start_end_point[0][0], start_end_point[0][1]],
        # goal=[start_end_point[1][0], start_end_point[1][1]])

        env_info = env.get_env()

        # lidar_state: 724 np.array
        lidar_self = np.array(env_info[0])
        # get lidar information or self state information using slice -----------------------
        # 1x724 np.array
        lidar_self = lidar_self[:][np.newaxis, :]

        # visual: 1x80x80 np.array
        visual = env_info[1][np.newaxis, :]

        done, reward = env_info[2], env_info[3]

        # initialize the first state
        for i in range(n_fram_stack):
            visual_deque.append(visual)
        # 4x80x80  np.array
        visual_state = np.concatenate((visual_deque[0], visual_deque[1], visual_deque[2], visual_deque[3]), axis=0)

        for i in range(n_fram_stack):
            state_deque.append(lidar_self)
        # 4x724  np.array
        lidar_self_state = np.concatenate((state_deque[0], state_deque[1], state_deque[2], state_deque[3]), axis=0)
        # 1x2896 np.array
        lidar_self_state = lidar_self_state.reshape(1, -1)

        while True:
            action = agent.choose_action_evaluate(visual_state, lidar_self_state)
            v_cmd = float(action[0][0].asnumpy())
            w_cmd = float(action[0][1].asnumpy())
            action = [v_cmd, w_cmd]
            episode_steps += 1

            env.step(action)
            env_info = env.get_env()
            lidar_self, visual, done, reward = env_info[0], env_info[1], env_info[2], env_info[3]
            episode_reward += reward

            # lidar information and self state information   using slice --------------------------------
            lidar_self = np.array(lidar_self)
            lidar_self = lidar_self[:][np.newaxis, :]

            visual = visual[np.newaxis, :]
            state_deque.append(lidar_self)
            visual_deque.append(visual)

            next_visual_state = np.concatenate((visual_deque[0], visual_deque[1], visual_deque[2], visual_deque[3]),
                                               axis=0)
            next_lidar_self_state = np.concatenate((state_deque[0], state_deque[1], state_deque[2], state_deque[3]),
                                                   axis=0)
            next_lidar_self_state = next_lidar_self_state.reshape(1, -1)

            visual_state, lidar_self_state = next_visual_state, next_lidar_self_state
            if done:
                break
            if episode_steps >= 300:
                break
        if episode_reward > target_reward:
            success_times += 1
        episode_reward_list.append(episode_reward)
        print('episode %d ends with reward %f' % (episode, episode_reward))
    plt.plot(episode_reward_list)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.show()















