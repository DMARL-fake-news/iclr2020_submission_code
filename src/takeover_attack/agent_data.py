
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

class agent_data(object):

    def __init__(self, agent_id):

        self.agent_id = agent_id

        self.act_obs = []
        self.rewards = []  
        self.act_taken = []

    def observe_act(self, act):

        self.act_obs.append(act)

    def observe_reward(self, rew):

        self.rewards.append(rew)

    def record_act_taken(self, act):

        self.act_taken.append(act)

    def reset(self, signal, q_agent, a_type):

        self.signal = signal
        self.q_agent = q_agent
        self.a_type = a_type
        self.act_obs = []
        self.act_taken = []
        self.rewards = []
    