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

import argparse
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import string
import pickle
import argparse
import pdb
from copy import deepcopy
import os
import pandas as pd
import git
from multiprocessing import Process
import multiprocessing as mult
import time

import q_agent as qa
import agent_data
from dqn_utils import *
import env_adv as envim

def RMSPropOptimizer(learning_rate=5e-4, momentum=0.05):

    return qa.OptimizerSpec(
        constructor=tf.train.RMSPropOptimizer,
        lr_schedule=ConstantSchedule(learning_rate),
        kwargs={'momentum' : momentum}
    )

def AdamOptimizer(learning_rate=1e-3):

    return qa.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        lr_schedule=ConstantSchedule(learning_rate),
        kwargs={}
    )

def piecewise_exploration_schedule(num_timesteps, limit_rate=0.02):
    return PiecewiseSchedule(
        [
            (0, 0.1),
            (num_timesteps * 0.1, limit_rate),
        ], outside_value=limit_rate
    )

def constant_exploration_schedule(constant_rate=0.05):
    return ConstantSchedule(constant_rate)

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def set_global_seeds(i):
    tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    session = tf.Session()
    return session

class logger(object):

    def __init__(self, average_window, print_every,
    log_file, agent_types):

        self.agent_types = agent_types

        self.episode_rewards = self.make_log_dict()
        self.terminal_rewards = self.make_log_dict()
        self.bellman_errors = self.make_log_dict()
        self.grad_norm_min_logs = self.make_log_dict()
        self.grad_norm_max_logs = self.make_log_dict()
        self.qval_trains_log = self.make_log_dict()
        self.qval_targets_log = self.make_log_dict()

        self.mean_episode_reward = self.make_log_dict()
        self.mean_terminal_reward = self.make_log_dict()
        self.mean_bellman_errors = self.make_log_dict()

        self.test_rewards = self.make_log_dict()
        self.test_action_signal_corr = self.make_log_dict()
        self.test_action_accuracy = self.make_log_dict()

        self.average_window = average_window
        self.print_every = print_every
        self.log_file = log_file

    def make_log_dict(self):

        d = {}
        for t in self.agent_types:
            d[t] = []

        return d

    def log_type(self, a_type, agents, errors, grad_norms, qval_trains, qval_targets,
    total_time, q_agents, iteration, args):

        agents = [a for a in agents if a.a_type == a_type]

        n_t_agents = len(agents)

        rewards = np.zeros((
            n_t_agents, args.T, args.n_batches
        ))

        error_a = np.zeros((
            n_t_agents, args.n_batches
        ))

        grad_norms_min = np.zeros((
            n_t_agents, 1
        ))

        grad_norms_max = np.zeros((
            n_t_agents, 1
        ))

        qval_trains_l = np.zeros((
            n_t_agents, 2
        ))

        qval_targets_l = np.zeros((
            n_t_agents, 2
        )) 

        for i, a in enumerate(agents):
            error_a[i, :] = errors[i]
            for t in range(args.T):
                rewards[i, t, :] = a.rewards[t]

            grad_norms_min[i, 0] = np.min(grad_norms[i])
            grad_norms_max[i, 0] = np.max(grad_norms[i])
            qval_trains_l[i, :] = qval_trains[i][0, :]
            qval_targets_l[i, :] = qval_targets[i][0, :]

        m_reward = np.mean(rewards)
        m_reward_terminal = np.mean(rewards[:, -1, :])
        m_errors = np.mean(error_a)

        self.episode_rewards[a_type].append(m_reward)
        self.terminal_rewards[a_type].append(m_reward_terminal)
        self.bellman_errors[a_type].append(m_errors)
        self.grad_norm_min_logs[a_type].append(grad_norms_min)
        self.grad_norm_max_logs[a_type].append(grad_norms_max)        
        self.qval_trains_log[a_type].append(qval_trains_l)
        self.qval_targets_log[a_type].append(qval_targets_l)

        if (
            len(self.episode_rewards[a_type]) % self.average_window == 0 and 
            len(self.episode_rewards[a_type]) != 0
        ):
            self.mean_episode_reward[a_type].append(
                np.mean(self.episode_rewards[a_type][-self.average_window:])
            )
            self.mean_terminal_reward[a_type].append(
                np.mean(self.terminal_rewards[a_type][-self.average_window:])
            )
            self.mean_bellman_errors[a_type].append(
                np.mean(self.bellman_errors[a_type][-self.average_window:])
            )

    def log(self, agents, errors, grad_norms, qval_trains, qval_targets,
    total_time, q_agents, iteration, args):
        
        a_types = list(set([a.a_type for a in agents]))

        for a_type in a_types:
            self.log_type(a_type, agents, errors, grad_norms, qval_trains, qval_targets,
            total_time, q_agents, iteration, args)

        if iteration % self.print_every == 0 and iteration != 0:
            
            if not isinstance(q_agents, dict):
                learning_rate = q_agents.optimizer_spec.lr_schedule.value(
            total_time)

                exploration_rate = q_agents.exploration.value(total_time)          
            else:
                learning_rate = list(q_agents.values())[0].optimizer_spec.lr_schedule.value(
                total_time)

                exploration_rate = list(q_agents.values())[0].exploration.value(total_time)

            print('Iteration %d' % iteration)
            for a_type in a_types:
                print('mean reward of type %s (%d episodes) %f' % (
                    a_type, self.average_window, self.mean_episode_reward[a_type][-1]))
                print('mean terminal reward of type %s (%d episodes) %f' % (
                    a_type, self.average_window, self.mean_terminal_reward[a_type][-1]
                ))
                print('mean bellman error of type %s (%d episodes) %f' % (
                    a_type, self.average_window, self.mean_bellman_errors[a_type][-1]
                ))
            print('Learning rate %f' % learning_rate)
            print('Exploration rate %f' % exploration_rate)

            with open(self.log_file, 'wb')  as f:
                pickle.dump([
                    self.episode_rewards, self.terminal_rewards, self.bellman_errors,
                    self.grad_norm_min_logs, self.grad_norm_max_logs, 
                    self.qval_trains_log, self.qval_targets_log,
                    self.test_rewards, self.test_action_signal_corr, self.test_action_accuracy
                ], f, pickle.HIGHEST_PROTOCOL)

    def test_log(self, test_reward, action_signal_correlation, accuracy):

        for a_type in test_reward:
            self.test_rewards[a_type].append(test_reward[a_type])
            self.test_action_signal_corr[a_type].append(
                action_signal_correlation[a_type]
            )
            self.test_action_accuracy[a_type].append(
                accuracy[a_type]
            )

def make_csv_log(rews, arec, env, save_path, fname):

    T = len(rews)
    n_agents = len(rews[0])
    n_batches = rews[0][0].shape[0]
    # Structure
    # t, Batch, Agent, Attack, Attack_mode, Signal, World, Action, Reward

    output = {
        'batch' : [],
        't' : [],
        'agent' : [],
        'world' : [],
        'attack_mode' : [],
        'attack' : [],
        'bias' : [],
        'signal' : [],
        'action' : [],
        'reward' : []
    }

    for t in range(T):
        for a in range(n_agents):
            for b in range(n_batches):
                if env.attack:
                    attack = 1 if a in env.attack_idx[b, :] else 0
                    attack_mode = 1
                else:
                    attack = 0
                    attack_mode = 0
                signal = env.signals[b, a]
                world = env.world[b, 0]
                bias = env.bias_per_agent if attack == 1 else 0
                action = arec[t][a][b]
                reward = rews[t][a][b]

                output['t'].append(t)
                output['batch'].append(b)
                output['agent'].append(a)
                output['world'].append(world)
                output['attack_mode'].append(attack_mode)
                output['attack'].append(attack)
                output['signal'].append(signal)
                output['action'].append(action)
                output['reward'].append(reward)
                output['bias'].append(bias)

    df = pd.DataFrame(output)
    df.sort_values(by=['batch', 't', 'agent'], inplace=True)
    df.to_csv(save_path + fname, index=False)


def run_test(sess, log, iteration, total_time, q_agents, agents,
env, args, save_path='', fname='', do_save=False):

    def print_test_summary(rews):

        rewards = []
        rewards_sd = []
        rewards_min = []
        rewards_max = []

        for rew in rews:
            rw = np.zeros((len(rew), len(rew[0])))
            for j in range(len(rew)):
                rw[j, :] = rew[j]
            rewards.append(np.mean(rw))
            rewards_sd.append(np.std(rw))
            rewards_min.append(np.min(rw))
            rewards_max.append(np.max(rw))

        mean_test_reward = np.mean(rewards)

        print('========================================')
        print('Testing at iteration %d' % iteration)
        print('Test terminal reward (%d batches) %f +- %f' %(
        args.n_batches, rewards[-1], rewards_sd[-1]))
        print('Test terminal reward (%d batches) max %f, min %f' %(
        args.n_batches, rewards_max[-1], rewards_min[-1]))
        print('Test total reward (%d batches) %f' %(
        args.n_batches, mean_test_reward))
        print('========================================')

        return rewards[-1]

    rews, arec, signals_rec, env_res = run_episode(
        sess, log, iteration, total_time, q_agents, agents, env, args, 
        test=True
    )

    rewards_last_T = {}
    action_signal_correlation = {}
    accuracy = {}

    for a_type in rews:
        print('========================================')
        print('Test for type %s' % a_type)
        rT = print_test_summary(rews[a_type])
        rewards_last_T[a_type] = rT

        ac = arec[a_type][0][0]
        si = signals_rec[a_type][0][0]
        action_signal_correlation[a_type] = np.corrcoef(
            ac, si
        )[0, 1]
        accuracy[a_type] = (ac == env_res.world[:, 0]).mean()
        

    if do_save:
        if args.attack:
            fname += '_attack_mode_n_attack_%d_bias_%f'%(args.n_attack, args.bias_per_agent)
        with open(save_path + fname, 'wb') as f:
            pickle.dump([rews, arec, signals_rec], f, pickle.HIGHEST_PROTOCOL)
        
        if args.eval_save_csv:
            make_csv_log(rews, arec, env, save_path, fname + '_full_log.csv')

    return rewards_last_T, action_signal_correlation, accuracy

def split_by_type(l, agents, is_signal=False):

    a_types = list(set([a.a_type for a in agents]))
    d = {}
    for t in a_types:
        d[t] = []

    for i, a in enumerate(agents):
        if is_signal:
            d[a.a_type].append(l[:, i])
        else:
            d[a.a_type].append(l[i])

    return d  

def run_episode(sess, log, iteration, total_time, q_agents, agents, env, args, 
test=False):

    obs = env.reset()

    for i, a in enumerate(agents):
        if env.attack:
            if i in env.attack_idx:
                a.reset(env.signals[:, i], q_agents['attacker'], 'attacker')
            else:
                a.reset(env.signals[:, i], q_agents['citizen'], 'citizen')
        else:
            a.reset(env.signals[:, i], q_agents['citizen'], 'citizen')

    a_types = list(set([a.a_type for a in agents]))

    def make_dict():
        d = {}
        for t in a_types:
            d[t] = []

        return d

    rews = make_dict()
    arec = make_dict()
    signals_rec = make_dict()

    for t in range(args.T):

        for i, a in enumerate(agents):
            a.observe_act(obs[i])

        actions = []
        for i, a in enumerate(agents):
            ids = [a.agent_id for j in range(args.n_batches)]
            action = a.q_agent.take_action(
                sess, a.act_obs,
                a.signal, ids, t,
                total_time, args.n_batches, test, env.world
            )
            if args.attack and args.random_attacker and a.a_type == 'attacker':
                action = np.random.randint(0, args.num_actions - 1, size=args.n_batches)
            actions.append(action)
            a.record_act_taken(action)

        obs, rewards = env.step(actions)
        signals = env.signals

        rewards_split_types = split_by_type(rewards, agents)
        actions_split_types = split_by_type(actions, agents)
        signals_split_types = split_by_type(signals, agents, is_signal=True)

        for a_type in a_types:
            rews[a_type].append(rewards_split_types[a_type])
            arec[a_type].append(actions_split_types[a_type])
            signals_rec[a_type].append(signals_split_types[a_type])
        
        for i, a in enumerate(agents):
            # NOTE: this is not used in learning! Only for record keeping.
            a.observe_reward(rewards[i])

    if not test:

        errors = []
        grad_norms = []
        qval_trains = []
        qval_targets = []

        for i, a in enumerate(agents):
            ids = [a.agent_id for j in range(args.n_batches)]
            error, grad_norm, qval_train, qval_target = a.q_agent.update(env,
                sess, a.act_obs, a.act_taken,
                a.signal, ids, a.rewards,
                total_time, iteration, env.world
            )
            errors.append(error)
            grad_norms.append(grad_norm)
            qval_trains.append(qval_train[0])
            qval_targets.append(qval_target[0])

        log.log(agents, errors, grad_norms, qval_trains, qval_trains,
        total_time, q_agents, iteration, args)

        total_time += args.T

        return total_time
    else:
        return rews, arec, signals_rec, env

def train_or_eval(args):

    RESTORE = False
    EVALUATE = False
    EVALUATE_SAVE_CSV = False

    if args.restore:
        RESTORE = args.restore
        restore_file = args.restore_file
        restore_id = args.restore_run_id
        EVALUATE = args.evaluate
        n_batch_eval = args.n_batches_evaluation
        EVALUATE_SAVE_CSV = args.eval_save_csv

    if not RESTORE:
        run_id = id_generator()
        log_file = args.log_path + run_id
        arg_file = args.log_path + 'args_' + run_id
        with open(arg_file, 'wb')  as f:
            pickle.dump(args, f, pickle.HIGHEST_PROTOCOL)
    else:
        run_id = restore_id
        log_file = args.log_path + run_id +'_tmp'
        arg_file = args.log_path + 'args_' + run_id
        ATTACK = args.attack
        N_ATTACK = args.n_attack
        BIAS_PER_AGENT = args.bias_per_agent
        EVAL_SAVE_PATH = args.eval_path
        SAVE_PATH = args.save_path
        LOG_PATH = args.log_path
        with open(arg_file, 'rb')  as f:
            args = pickle.load(f)
        if ATTACK:
            print('Setting attack parameters!')
            args.attack = ATTACK
            args.n_attack = N_ATTACK
            args.bias_per_agent = BIAS_PER_AGENT
        else:
            args.attack = ATTACK
        args.eval_save_csv = EVALUATE_SAVE_CSV
        args.eval_path = EVAL_SAVE_PATH
        args.save_path = SAVE_PATH
        args.log_path = LOG_PATH

    args.network_path = osp.join(
        osp.abspath(osp.join('../..')),
        args.network_path)

    print(args.network_file)

    args.save_path = osp.join(
        args.save_path, run_id)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    env = envim.env_adv(args)
    args_test = deepcopy(args)
    if not EVALUATE:
        args_test.n_batches = args_test.n_batches_test
    else:
        args_test.n_batches = n_batch_eval

    env_test = envim.env_adv(args_test)

    log = logger(args.average_window, args.print_every, log_file, ['citizen', 'attacker'])

    sess = get_session()
    
    set_global_seeds(args.seed)

    if args.optimizer == 'RMSProp':
        my_optimizer = RMSPropOptimizer(learning_rate=args.learning_rate,
        momentum=args.momentum)
    elif args.optimizer == 'Adam':
        my_optimizer = AdamOptimizer(learning_rate=args.learning_rate)
    else:
        raise ValueError(
            'Must provide a valid optimizer.'
        )

    if args.exploration_schedule == 'constant':
        my_exploration_schedule = constant_exploration_schedule(
            constant_rate=args.exploration_rate)
    elif args.exploration_schedule == 'piece_wise_linear':
        my_exploration_schedule = piecewise_exploration_schedule(900000, 
        limit_rate=args.exploration_rate)
    else:
        raise ValueError(
            'Must provide a valid exploration schedule.'
        )

    q_agents = {}
    q_agents['citizen'] = qa.q_agent(
    args.num_actions, args.n_agents, args.T, args.gamma,
    my_optimizer,
    my_exploration_schedule,
    args.target_reset,
    args.n_hidden,
    'q_func_train_citizen',
    'q_func_target_citizen',
    grad_norm_clipping=args.grad_norm_clipping, reuse=False, double_q=args.double_q,
    n_layers=args.n_layers)

    q_agents['attacker'] = qa.q_agent(
    args.num_actions, args.n_agents, args.T, args.gamma,
    my_optimizer,
    my_exploration_schedule,
    args.target_reset,
    args.n_hidden,
    'q_func_train_attack',
    'q_func_target_attack',
    grad_norm_clipping=args.grad_norm_clipping, reuse=False, double_q=args.double_q,
    n_layers=args.n_layers)

    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    saver = tf.train.Saver(max_to_keep=args.max_to_keep)

    agents = []
    for i in range(args.n_agents):
        agents.append(agent_data.agent_data(i))

    sess.run(tf.initializers.global_variables())

    best_n_rewards = [0]
    if RESTORE:
        restore_path = osp.join(args.save_path, restore_file)
        saver.restore(sess, restore_path)

    if not EVALUATE:
    
        total_time = 0
        for it in range(args.n_iterations):
            total_time = run_episode(
                sess, log, it, total_time, q_agents, agents, env, args
            )
            sess.run(global_step.assign_add(1))
            if not args.save_regularly:
                if (it + 1) % 100 == 0:
                    test_reward, action_signal_correlation, accuracy = run_test(
                        sess, log, it, total_time, q_agents, agents,
                        env_test, args_test
                    )
                    log.test_log(test_reward, action_signal_correlation, accuracy)
                    if test_reward['citizen'] > np.min(best_n_rewards):
                        if len(best_n_rewards) < args.max_to_keep:
                            best_n_rewards.append(test_reward['citizen'])
                        else:
                            min_r = np.min(best_n_rewards)
                            best_n_rewards = [x for x in best_n_rewards if x != min_r]
                            best_n_rewards.append(test_reward['citizen'])
                        saver.save(sess,
                        osp.join(args.save_path, run_id + '_sess_chkpt_best_reward_%f'%test_reward['citizen']),
                        global_step=global_step)
                    print('Current best reward = %s' % best_n_rewards)
            else:
                if (it + 1) % args.test_freq == 0:
                    test_reward, action_signal_correlation, accuracy = run_test(
                        sess, log, it, total_time, q_agents, agents,
                        env_test, args_test
                    )
                    log.test_log(test_reward, action_signal_correlation, accuracy)
                if (it + 1) % args.checkpt_freq == 0:
                    saver.save(sess,
                        osp.join(args.save_path, run_id + '_sess_chkpt_iteration_%d'%it),
                        global_step=global_step)

    
    else:
        eval_save_path = args.eval_path

        fname = run_id + '_eval'
        test_reward, _, _ = run_test(
                sess, log, 0, 0, q_agents, agents,
                env_test, args_test, save_path=eval_save_path,
                fname=fname, do_save=True
            )
        for a_type in test_reward:
            print('evaluating, type %s, reward = %f' % (a_type, test_reward[a_type]))



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # NETWORK ARCHITECTURE PARAMETERS
    parser.add_argument('-n_hidden', help='GRU and fully connected hidden units', type=int, default=64)
    parser.add_argument('-n_layers', help='number of GRU layers', type=int, default=2)

    # LEARNING PARAMETERS
    parser.add_argument('-seed', help='global seed', type=int, default=5)
    parser.add_argument('-learning_rate', help='learning rate', type=float, default=5e-4)
    parser.add_argument('-momentum', help='momentum for RMSProp', type=float, default=0.05) 
    parser.add_argument('-optimizer', help='which optimizer to use', type=str, default='RMSProp') 
    parser.add_argument('-exploration_rate', help='exploration rate', type=float, default=0.05)
    parser.add_argument('-exploration_schedule', help='exploration schedule to use',
    type=str, default='constant')
    parser.add_argument('-grad_norm_clipping', help='max gradient norm to clip to', type=float, default=10.)
    parser.add_argument('-target_reset', help='dqn reset frequency', type=int, default=100)
    parser.add_argument('-n_iterations', help='number of training episodes', type=int, default=12000)
    parser.add_argument('-n_batches', help='number of batches', type=int, default=256)
    parser.add_argument('-n_batches_test', help='number of batches for testing during training',
    type=int, default=2000)
    parser.add_argument('-n_batches_evaluation',
    help='number of batches for testing during evaluation post traing',type=int, default=10000)
    parser.add_argument('-double_q', help='do doubel q learning', action="store_true")
    parser.add_argument('-n_experiments', help='number of experiments to run in parallel', type=int, default=1)

    # GAME PARAMETERS
    parser.add_argument('-num_actions', help='number of actions', type=int, default=2)
    parser.add_argument('-n_states', help='number of states', type=int, default=2)
    parser.add_argument('-n_agents', help='number of agents', type=int, default=10)
    parser.add_argument('-var', help='variance of signal', type=float, default=1.)
    parser.add_argument('-T', help='number of time steps', type=int, default=10)
    parser.add_argument('-gamma', help='discount factor', type=float, default=0.99)
    parser.add_argument('-network_path', help='location of network file', default='social_network_files/')
    parser.add_argument('-network_file', help='network file', default='social_network_complete.txt')
    parser.add_argument('-attack', help='whether to run in attack mode', action="store_true")
    parser.add_argument('-n_attack', help='number of nodes to attack', type=int, default=1)
    parser.add_argument('-bias_per_agent', help='attack bias per agent', type=float, default=3.)
    parser.add_argument('-random_attacker', help='if random attacker should be used instead of trained attacker', action="store_true")

    # LOGGING PARAMTERS
    parser.add_argument('-print_every', help='reward printing interval', type=int, default=100)
    parser.add_argument('-average_window', help='reward averaging interval', type=int, default=100)
    parser.add_argument('-log_path', help='location of log file', default='log/')
    parser.add_argument('-save_path', help='location of model checkpoints', default='checkpoints/')
    parser.add_argument('-max_to_keep', help='number of models to save', type=int, default=5)
    parser.add_argument('-code_version', help='git version of code', default='no_version')
    parser.add_argument('-experiment_name', help='name of experiment for easy identification', default='no_name')
    parser.add_argument('-save_regularly', help='whether to save model checkpoints at regular intervals', action='store_true')
    parser.add_argument('-checkpt_freq', help='frequency of model checkpoints under regular saving', type=int, default=100)
    parser.add_argument('-test_freq', help='frequency to run test', type=int, default=5)
    
    # RESTORE AND EVALUATE PARAMETERS
    parser.add_argument('-restore', help='restore best saved model', action="store_true")
    parser.add_argument('-restore_run_id', help='id of run to restore', type=str, default='')
    parser.add_argument('-restore_file', help='file of run to restore', type=str, default='')
    parser.add_argument('-evaluate', help='evaluate a saved model', action="store_true")
    parser.add_argument('-eval_path', help='location of evaluation output', default='evaluation/')
    parser.add_argument('-eval_save_csv', help='save evaluated model output as csv', action="store_true")
    parser.add_argument('-time_stamp', help='time stamp to recover', default='no_time_stamp')
    parser.add_argument('-load_old_format', help='load files in old folder structure', action="store_true")
    
    args = parser.parse_args()

    # add code version number

    try:
        repo = git.Repo(search_parent_directories=True)
        args.code_version = repo.head.object.hexsha
    except:
        args.code_version = '0'

    print('===================================================')
    print('USING PARAMETERS')
    print(args)
    print('===================================================')

    if not args.evaluate:

        if not args.restore:
            exp_path = args.experiment_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")

            args.save_path = osp.join(
            osp.abspath(osp.join('../..')), 'results', exp_path,
            args.save_path)

            args.log_path = osp.join(
            osp.abspath(osp.join('../..')), 'results', exp_path,
            args.log_path)

            if not os.path.exists(args.log_path):
                os.makedirs(args.log_path)

            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)

        processes = []

        args.n_experiments = min(args.n_experiments, max(mult.cpu_count() - 1, 1))
        print('Running a total of %d experiments in parallel.' % args.n_experiments)

        for e in range(args.n_experiments):
            seed = args.seed + 10*e
            args_run = deepcopy(args)
            args_run.seed = seed
            print('Running experiment with seed %d'%seed)

            def train_func():
                train_or_eval(args_run)

            p = Process(target=train_func, args=tuple())
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    else:

        if not args.load_old_format:
            exp_path = args.experiment_name 

            args.save_path = osp.join(
            osp.abspath(osp.join('../..')), 'results', exp_path,
            args.save_path)

            args.log_path = osp.join(
            osp.abspath(osp.join('../..')), 'results', exp_path,
            args.log_path)

            args.eval_path = osp.join(
            osp.abspath(osp.join('../..')), 'results', exp_path,
            args.eval_path)

            if not os.path.exists(args.eval_path):
                os.makedirs(args.eval_path)

        else:
            
            args.save_path = osp.join(
            osp.abspath(osp.join('../..')), 
            args.save_path)

            args.log_path = osp.join(
            osp.abspath(osp.join('../..')), 
            args.log_path)

            args.eval_path = osp.join(
            osp.abspath(osp.join('../..')),
            args.eval_path)

            if not os.path.exists(args.eval_path):
                os.makedirs(args.eval_path)


        train_or_eval(args)

    

