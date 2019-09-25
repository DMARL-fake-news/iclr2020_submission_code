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

class env_adv():

    def __init__(self, args):
        
        self.T = args.T
        self.var = args.var
        self.n_states = args.n_states
        self.n_batches = args.n_batches
        self.n_agents = args.n_agents
        network = self.load_network(args.network_path + args.network_file)
        self.neighborhoods = []
        for j in range(self.n_agents):
            neighborhood = network[j]
            self.neighborhoods.append(
                [neighborhood[i] for i in range(len(neighborhood))]
                )

        try:
            self.attack = args.attack
            self.n_attack = args.n_attack
        except AttributeError:
            print('Setting attack parameters to default')
            self.attack = False
            
        self.reset()

    def reset(self):
        
        self.time_step = 0
        self.world = np.random.randint(
            0, self.n_states, size=(self.n_batches, 1)
            )
        errs = np.random.normal(
            size=(self.n_batches, self.n_agents)
            ) * np.sqrt(self.var)
        
        self.signals = errs + self.world

        if self.attack:
            self.attack_idx = np.random.choice(
                    np.arange(0, self.n_agents), size=self.n_attack, replace=False)
            self.signals[:, self.attack_idx] = errs[:, self.attack_idx] * 0.0 + self.world
        
        tmp = np.zeros((self.n_batches, ), dtype=int)
        self.last_actions = [tmp for i in range(self.n_agents)]

        agent_observations = self.make_agent_obs(self.last_actions)

        return agent_observations

    def make_agent_obs(self, actions):

        agent_observations = []
        for i in range(self.n_agents):
            observed_actions = np.zeros((self.n_batches, self.n_agents))
            for j in range(self.n_agents):
                if j in self.neighborhoods[i] or i == j:
                    observed_actions[:, j] = actions[j] + 1
            agent_observations.append(observed_actions)

        return agent_observations


    def step(self, actions):
        
        self.last_actions = actions

        rewards = []
        rewards_array = np.zeros((self.n_batches, self.n_agents))
        
        for i, action in enumerate(actions):
        
            reward = np.zeros((self.n_batches, ))
            reward[action == self.world[:, 0]] = 1
            rewards_array[:, i] = reward

            rewards.append(reward)

        if self.attack:
            non_attacking_rewards = (
                rewards_array.sum(axis=1) - rewards_array[:, self.attack_idx].sum(axis=1)
            )
            for i in self.attack_idx:
                rewards[i] = -non_attacking_rewards / self.n_agents
            
        self.time_step += 1

        agent_observations = self.make_agent_obs(self.last_actions)
        
        return agent_observations, rewards

    def load_network(self, filename):

        f = open(filename, 'r')
        neighborhoods = dict() 
        cnt = 0
        for l in f:
            tmp = [int(i) for i in l.split(',')]
            neighborhoods[cnt] = tmp[1:]
            cnt += 1

        return neighborhoods
