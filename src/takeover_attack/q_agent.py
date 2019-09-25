# Copyright 2019
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import pdb
import random
import numpy as np
from collections import namedtuple
from dqn_utils import *

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

class model(object):

    def __init__(self, obs, n_hidden, num_actions, scope, 
    time_steps,
    reuse=False):

        self.n_hidden = n_hidden
        self.obs_stack = tf.stack(obs, axis=0)
        self.time_steps = time_steps
        self.batch_size = tf.shape(self.obs_stack)[1]
        self.qvals = []

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.create_rnn(self.obs_stack)
            self.output_r = tf.reshape(self.output, [self.time_steps * self.batch_size, -1])
            self.output_r = tf.reshape(self.output,
            [-1, self.output.shape[2]])

            self.linear_1 = tf.layers.dense(self.output_r, units=self.n_hidden[0],activation=tf.nn.relu)
            self.qvals_ = tf.layers.dense(self.linear_1, units=num_actions,
            activation=None)
            self.q_vals_t = tf.reshape(self.qvals_, [self.time_steps, self.batch_size, -1])
            self.qvals = tf.unstack(self.q_vals_t)

    def create_rnn(self, obs):
        layers = [tf.nn.rnn_cell.GRUCell(size) for size in self.n_hidden]
        cells = tf.nn.rnn_cell.MultiRNNCell(layers)
        zero_states = cells.zero_state(self.batch_size, dtype=tf.float32)
        self.in_state = tuple([tf.placeholder_with_default(state, [None, state.shape[1]]) 
                                for state in zero_states])
        self.output, self.out_state = tf.nn.dynamic_rnn(cells, obs,
                                                        initial_state=self.in_state,
                                                        time_major=True)

class q_agent(object):

    def __init__(self, num_actions, n_agents, time_steps, gamma,
    optimizer_spec, exploration, target_update_freq,
    n_hidden, scope_train, scope_target, batch_size=64,
    grad_norm_clipping=10, double_q=False, n_layers=1, reuse=False):

        self.optimizer_spec = optimizer_spec
        self.exploration = exploration
        self.target_update_freq = target_update_freq
        
        self.num_actions = num_actions
        self.n_agents = n_agents
        self.time_steps = time_steps

        self.agent_id_ph = tf.placeholder(tf.uint8, [None], name='agent_id')
        self.agent_id = self.agent_id_encoding(self.agent_id_ph)
        self.signal_ph = tf.placeholder(tf.float32, [None], name='signal')

        self.obs = [] # all info available at t
        self.act_obs = [] # observed actions from others at t (actions taken in t-1)
        self.rewards = [] # rewards for agent at t
        self.acts = [] # action taken by agent at t

        for t in range(time_steps):
            act_obs = tf.placeholder(tf.uint8, [None, n_agents],
            name='act_obs_t%d'%t)
            self.act_obs.append(act_obs)

            act_obs_enc = self.action_encoding(act_obs)
            
            self.obs.append(
                    tf.concat(
                    [act_obs_enc, self.agent_id,
                    tf.expand_dims(self.signal_ph, 1)], 1
                )
            )

            reward = tf.placeholder(tf.float32, [None], name='reward_t%d'%t)
            self.rewards.append(reward)

            act = tf.placeholder(tf.int32, [None], name='act_t%d'%t)
            self.acts.append(act)

        self.q_train = model(self.obs, n_layers * [n_hidden], num_actions,
        scope_train, time_steps, reuse=reuse)
        self.q_target = model(self.obs, n_layers * [n_hidden], num_actions,
        scope_target, time_steps, reuse=reuse)

        self.targets = []
        self.q_trains = []

        for t in range(time_steps):
            
            if t < time_steps - 1:
                if not double_q:
                    self.y = self.rewards[t] + gamma * tf.reduce_max(
                        self.q_target.qvals[t+1], axis=1)
                else:
                    self.amax = tf.cast(
                        tf.argmax(self.q_train.qvals[t+1], axis=1),
                        tf.int32)
                    self.idx = tf.stack([tf.range(tf.shape(self.amax)[0]), self.amax], 1)                    
                    self.qv_target_max = tf.gather_nd(
                        self.q_target.qvals[t+1], self.idx)
                    self.y = self.rewards[t] + gamma * self.qv_target_max
            else:
                self.y = self.rewards[t]

            indices = tf.stack(
                [tf.range(tf.shape(self.acts[t])[0]), self.acts[t]], 1)
            qv_train_a = tf.gather_nd(self.q_train.qvals[t], indices)
            self.targets.append(self.y)
            self.q_trains.append(qv_train_a)
            
        T = self.time_steps
        batch_size = tf.shape(self.obs[0])[0]
        self.targets_s = tf.reshape(tf.stack(self.targets, axis=0), [T * batch_size, -1])
        self.q_trains_s = tf.reshape(tf.stack(self.q_trains, axis=0), [T * batch_size, -1])
        
        self.total_error = tf.losses.huber_loss(
            self.targets_s, self.q_trains_s)

        self.learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
        optimizer = self.optimizer_spec.constructor(learning_rate=self.learning_rate, **self.optimizer_spec.kwargs)

        q_train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_train)
        q_farget_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_target)
        self.train_fn, self.grads_norms, self.grad_norms_post_clip = minimize_and_clip(optimizer, self.total_error,
                 var_list=q_train_vars, clip_val=grad_norm_clipping)

        update_target_fn = []
        for var, var_target in zip(sorted(q_train_vars,        key=lambda v: v.name),
                                sorted(q_farget_vars, key=lambda v: v.name)):
            update_target_fn.append(var_target.assign(var))

        self.update_target_fn = tf.group(*update_target_fn)

        update_target_fn_soft = []
        theta = 0.01**(1./5000)
        for var, var_target in zip(sorted(q_train_vars,        key=lambda v: v.name),
                                sorted(q_farget_vars, key=lambda v: v.name)):
            update_target_fn_soft.append(var_target.assign(
                theta * var_target + (1-theta) * var))
        self.target_update_freq_soft = tf.group(*update_target_fn_soft)


    def take_action(self, sess, act_obs, signal,
    agent_id, episode_time, total_time, batch_size, test, world):

        eps = self.exploration.value(total_time) if not test else 0
        
        if random.uniform(0, 1) < eps:
            action = np.random.randint(0, self.num_actions - 1, size=batch_size)
        else:
            feed = {}
            for t in range(self.time_steps):
                if t < episode_time + 1:
                    feed[self.act_obs[t]] = act_obs[t]
                else:
                    feed[self.act_obs[t]] = act_obs[0]
            
            feed[self.signal_ph] = signal
            feed[self.agent_id_ph] = agent_id

            qvals = sess.run(
                self.q_train.qvals[episode_time],
                feed_dict = feed
            )
            
            action = np.argmax(qvals, axis=1)

        return action

    def update(self, env, sess, act_obs, act_taken, signal,
    agent_id, rewards, total_time, iteration, world, is_updater=False):

        feed = {}
        for t in range(self.time_steps):
            feed[self.act_obs[t]] = act_obs[t]
        for t in range(self.time_steps):
            feed[self.rewards[t]] = rewards[t]
            feed[self.acts[t]] = act_taken[t]

        feed[self.signal_ph] = signal
        feed[self.agent_id_ph] = agent_id
        feed[self.learning_rate] = self.optimizer_spec.lr_schedule.value(
            total_time)

        error, grads_norms, grad_norms_post_clip, _, targets, obs, qvta, qvtr = sess.run(
            [self.total_error, self.grads_norms, self.grad_norms_post_clip,
            self.train_fn, self.targets, self.obs,
            self.q_target.qvals, self.q_train.qvals
            ],
            feed_dict = feed
        )

        if iteration % self.target_update_freq == 0 and iteration !=0:
            print('resetting, iteration %d'%iteration)   
            sess.run(
                self.update_target_fn
            )

            print(error)
            print(grads_norms)
            print(grad_norms_post_clip)
            print(qvta[0][0, :])
            print(qvtr[0][0, :])
            print(world[0])
            print(qvta[5][0, :])
            print(qvtr[5][0, :])
            print(qvta[1][0, :])
            print(qvtr[1][0, :])
            print(obs[0][0, -1])

        return error, grad_norms_post_clip, qvtr, qvta

    def action_encoding(self, act_obs):

        ac_encoding = tf.one_hot(act_obs, self.num_actions + 1, axis=-1)
        ac_encoding = tf.reshape(
                ac_encoding,
                [-1, ac_encoding.shape[1] * ac_encoding.shape[2]]
                )

        return ac_encoding

    def agent_id_encoding(self, id_obs):

        id_encoding = tf.one_hot(id_obs, self.n_agents, axis=-1)
        
        return id_encoding




