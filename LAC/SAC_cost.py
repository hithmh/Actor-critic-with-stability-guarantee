import tensorflow as tf
import numpy as np
import time
from .squash_bijector import SquashBijector
from .utils import evaluate_training_rollouts
import tensorflow_probability as tfp
from collections import OrderedDict, deque
import os
from copy import deepcopy
import sys
sys.path.append("..")
from disturber.disturber import Disturber
from robustness_eval import training_evaluation
from pool.pool import Pool
import logger
from variant import *

SCALE_DIAG_MIN_MAX = (-20, 2)
SCALE_lambda_MIN_MAX = (0, 20)


class SAC_cost(object):
    def __init__(self,
                 a_dim,
                 s_dim,

                 variant,

                 action_prior = 'uniform',
                 ):



        ###############################  Model parameters  ####################################
        # self.memory_capacity = variant['memory_capacity']

        self.batch_size = variant['batch_size']
        gamma=variant['gamma']
        tau = variant['tau']
        self.approx_value = True if 'approx_value' not in variant.keys() else variant['approx_value']
        # self.memory = np.zeros((self.memory_capacity, s_dim * 2 + a_dim+ d_dim + 3), dtype=np.float32)
        # self.pointer = 0
        self.sess = tf.Session()
        self._action_prior = action_prior
        self.a_dim, self.s_dim = a_dim, s_dim
        target_entropy = variant['target_entropy']
        if target_entropy is None:
            self.target_entropy = -self.a_dim   #lower bound of the policy entropy
        else:
            self.target_entropy = target_entropy
        with tf.variable_scope('Actor'):
            self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
            self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')

            self.a_input = tf.placeholder(tf.float32, [None, a_dim], 'a_input')
            self.a_input_ = tf.placeholder(tf.float32, [None, a_dim], 'a_input_')
            self.R = tf.placeholder(tf.float32, [None, 1], 'r')

            self.terminal = tf.placeholder(tf.float32, [None, 1], 'terminal')
            self.LR_A = tf.placeholder(tf.float32, None, 'LR_A')
            self.LR_C = tf.placeholder(tf.float32, None, 'LR_C')
            self.LR_L = tf.placeholder(tf.float32, None, 'LR_L')
            # self.labda = tf.placeholder(tf.float32, None, 'Lambda')
            labda = variant['labda']
            alpha = variant['alpha']
            alpha3 = variant['alpha3']
            log_labda = tf.get_variable('lambda', None, tf.float32, initializer=tf.log(labda))
            log_alpha = tf.get_variable('alpha', None, tf.float32, initializer=tf.log(alpha))  # Entropy Temperature
            self.labda = tf.clip_by_value(tf.exp(log_labda), *SCALE_lambda_MIN_MAX)
            self.alpha = tf.exp(log_alpha)

            self.a, self.deterministic_a, self.a_dist = self._build_a(self.S, )  # 这个网络用于及时更新参数
            self.q1 = self._build_c(self.S, self.a_input, 'critic1')  # 这个网络是用于及时更新参数
            self.q2 = self._build_c(self.S, self.a_input, 'critic2')  # 这个网络是用于及时更新参数
            self.l = self._build_l(self.S, self.a_input)   # lyapunov 网络

            self.q1_a = self._build_c(self.S, self.a, 'critic1', reuse=True)
            self.q2_a = self._build_c(self.S, self.a, 'critic2', reuse=True)

            self.use_lyapunov = variant['use_lyapunov']
            self.adaptive_alpha = variant['adaptive_alpha']

            a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor/actor')
            c1_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor/critic1')
            c2_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor/critic2')
            l_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor/Lyapunov')

            ###############################  Model Learning Setting  ####################################
            ema = tf.train.ExponentialMovingAverage(decay=1 - tau)  # soft replacement

            def ema_getter(getter, name, *args, **kwargs):
                return ema.average(getter(name, *args, **kwargs))
            target_update = [ema.apply(a_params), ema.apply(c1_params),ema.apply(c2_params), ema.apply(l_params)]  # soft update operation

            # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
            a_, _, a_dist_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)  # replaced target parameters
            lya_a_, _, lya_a_dist_ = self._build_a(self.S_, reuse=True)
            # self.cons_a_input_ = tf.placeholder(tf.float32, [None, a_dim, 'cons_a_input_'])
            # self.log_pis = log_pis = self.a_dist.log_prob(self.a)
            self.log_pis = log_pis = self.a_dist.log_prob(self.a)
            self.prob = tf.reduce_mean(self.a_dist.prob(self.a))

            # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
            q1_ = self._build_c(self.S_, a_,'critic1', reuse=True, custom_getter=ema_getter)
            q2_ = self._build_c(self.S_, a_, 'critic2', reuse=True, custom_getter=ema_getter)
            l_ = self._build_l(self.S_, a_, reuse=True, custom_getter=ema_getter)
            self.l_ = self._build_l(self.S_, lya_a_, reuse=True)

            # lyapunov constraint
            self.l_derta = tf.reduce_mean(self.l_ - self.l + (alpha3+1) * self.R)

            labda_loss = -tf.reduce_mean(log_labda * self.l_derta)
            alpha_loss = -tf.reduce_mean(log_alpha * tf.stop_gradient(log_pis + self.target_entropy))
            self.alpha_train = tf.train.AdamOptimizer(self.LR_A).minimize(alpha_loss, var_list=log_alpha)
            self.lambda_train = tf.train.AdamOptimizer(self.LR_A).minimize(labda_loss, var_list=log_labda)

            if self._action_prior == 'normal':
                policy_prior = tf.contrib.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(self.a_dim),
                    scale_diag=tf.ones(self.a_dim))
                policy_prior_log_probs = policy_prior.log_prob(self.a)
            elif self._action_prior == 'uniform':
                policy_prior_log_probs = 0.0

            min_Q_target = tf.reduce_max((self.q1_a, self.q2_a), axis=0)
            self.a_preloss = a_preloss = tf.reduce_mean(self.alpha * log_pis + min_Q_target - policy_prior_log_probs)
            if self.use_lyapunov is True:
                a_loss = self.labda * self.l_derta + self.alpha * log_pis - policy_prior_log_probs
            else:
                a_loss = a_preloss

            self.a_loss = a_loss
            self.atrain = tf.train.AdamOptimizer(self.LR_A).minimize(a_loss, var_list=a_params)

            next_log_pis = a_dist_.log_prob(a_)
            with tf.control_dependencies(target_update):  # soft replacement happened at here
                min_next_q = tf.reduce_max([q1_, q2_], axis=0)
                q1_target = self.R + gamma * (1 - self.terminal) * tf.stop_gradient(
                    min_next_q - self.alpha * next_log_pis)  # ddpg
                q2_target = self.R + gamma * (1 - self.terminal) * tf.stop_gradient(
                    min_next_q - self.alpha * next_log_pis)  # ddpg
                if self.approx_value:
                    l_target = self.R + gamma * (1-self.terminal)*tf.stop_gradient(l_)  # Lyapunov critic - self.alpha * next_log_pis
                else:
                    l_target = self.R

                self.td_error1 = tf.losses.mean_squared_error(labels=q1_target, predictions=self.q1)
                self.td_error2 = tf.losses.mean_squared_error(labels=q2_target, predictions=self.q2)
                self.l_error = tf.losses.mean_squared_error(labels=l_target, predictions=self.l)
                self.ctrain1 = tf.train.AdamOptimizer(self.LR_C).minimize(self.td_error1, var_list=c1_params)
                self.ctrain2 = tf.train.AdamOptimizer(self.LR_C).minimize(self.td_error2, var_list=c2_params)
                self.ltrain = tf.train.AdamOptimizer(self.LR_L).minimize(self.l_error, var_list=l_params)

            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            self.diagnotics = [self.labda, self.alpha, self.td_error1, self.td_error2, self.l_error, tf.reduce_mean(-self.log_pis), self.a_loss]

            if self.use_lyapunov is True:
                self.opt = [self.ltrain, self.lambda_train]
            else:
                self.opt = [self.ctrain1, self.ctrain2, ]
            self.opt.append(self.atrain)
            if self.adaptive_alpha is True:
                self.opt.append(self.alpha_train)

    def choose_action(self, s, evaluation = False):
        if evaluation is True:
            return self.sess.run(self.deterministic_a, {self.S: s[np.newaxis, :]})[0]
        else:
            return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self, LR_A, LR_C, LR_L, batch):
        # if self.pointer>=self.memory_capacity:
        #     indices = np.random.choice(self.memory_capacity, size=self.batch_size)
        # else:
        #     indices = np.random.choice(self.pointer, size=self.batch_size)
        # bt = self.memory[indices, :]
        # bs = bt[:, :self.s_dim]  # state
        # ba = bt[:, self.s_dim: self.s_dim + self.a_dim]  # action
        # bd = bt[:, self.s_dim + self.a_dim: self.s_dim + 2*self.a_dim]  # action
        # br = bt[:, -self.s_dim - 3: -self.s_dim - 2]  # reward
        #
        # bterminal = bt[:, -self.s_dim - 1: -self.s_dim]
        #
        # bs_ = bt[:, -self.s_dim:]  # next state
        bs = batch['s']  # state
        ba = batch['a']  # action
        br = batch['r']  # reward

        bterminal = batch['terminal']
        bs_ = batch['s_']  # next state

        feed_dict = {self.a_input: ba, self.S: bs, self.S_: bs_, self.R: br, self.terminal: bterminal,
                     self.LR_C: LR_C, self.LR_A: LR_A, self.LR_L: LR_L}

        self.sess.run(self.opt, feed_dict)
        labda, alpha, q1_error, q2_error, l_error, entropy, a_loss = self.sess.run(self.diagnotics, feed_dict)

        return labda, alpha, q1_error, q2_error, l_error, entropy, a_loss

    def store_transition(self, s, a,d, r, l_r, terminal, s_):
        transition = np.hstack((s, a, d, [r], [l_r], [terminal], s_))
        index = self.pointer % self.memory_capacity  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, name='actor', reuse=None, custom_getter=None):
        if reuse is None:
            trainable = True
        else:
            trainable = False

        with tf.variable_scope(name, reuse=reuse, custom_getter=custom_getter):
            batch_size = tf.shape(s)[0]
            squash_bijector = (SquashBijector())
            base_distribution = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(self.a_dim), scale_diag=tf.ones(self.a_dim))
            epsilon = base_distribution.sample(batch_size)
            ## Construct the feedforward action
            net_0 = tf.layers.dense(s, 256, activation=tf.nn.relu, name='l1', trainable=trainable)#原始是30
            net_1 = tf.layers.dense(net_0, 256, activation=tf.nn.relu, name='l4', trainable=trainable)  # 原始是30
            mu = tf.layers.dense(net_1, self.a_dim, activation= None, name='a', trainable=trainable)
            log_sigma = tf.layers.dense(net_1, self.a_dim, None, trainable=trainable)
            log_sigma = tf.clip_by_value(log_sigma, *SCALE_DIAG_MIN_MAX)
            sigma = tf.exp(log_sigma)


            bijector = tfp.bijectors.Affine(shift=mu, scale_diag=sigma)
            raw_action = bijector.forward(epsilon)
            clipped_a = squash_bijector.forward(raw_action)

            ## Construct the distribution
            bijector = tfp.bijectors.Chain((
                squash_bijector,
                tfp.bijectors.Affine(
                    shift=mu,
                    scale_diag=sigma),
            ))
            distribution = tfp.distributions.ConditionalTransformedDistribution(
                    distribution=base_distribution,
                    bijector=bijector)

            clipped_mu = squash_bijector.forward(mu)

        return clipped_a, clipped_mu, distribution

    #critic模块
    def _build_c(self, s, a, name ='Critic', reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope(name, reuse=reuse, custom_getter=custom_getter):
            n_l1 = 256#30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net_0 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net_1 = tf.layers.dense(net_0, 256, activation=tf.nn.relu, name='l2', trainable=trainable)  # 原始是30
            return tf.layers.dense(net_1, 1, trainable=trainable)  # Q(s,a)

    def _build_l(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Lyapunov', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 256#30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net_0 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net_1 = tf.layers.dense(net_0, 256, activation=tf.nn.relu, name='l2', trainable=trainable)  # 原始是30
            return tf.layers.dense(net_1, 1, trainable=trainable)  # Q(s,a)

    def evaluate_value(self, s, a):

        return self.sess.run(tf.reduce_max((self.q1, self.q2), axis=0), {self.S: s[np.newaxis, :], self.a_input: a[np.newaxis, :]})[0]

    def save_result(self, path):

        save_path = self.saver.save(self.sess, path + "/policy/model.ckpt")
        print("Save to path: ", save_path)

    def restore(self, path):
        model_file = tf.train.latest_checkpoint(path+'/')
        if model_file is None:
            success_load = False
            return success_load
        self.saver.restore(self.sess, model_file)
        success_load = True
        return success_load


def train(variant):
    env_name = variant['env_name']
    env = get_env_from_name(env_name)
    env_params = variant['env_params']

    max_episodes = env_params['max_episodes']
    max_ep_steps = env_params['max_ep_steps']
    max_global_steps = env_params['max_global_steps']
    store_last_n_paths = variant['num_of_training_paths']
    evaluation_frequency = variant['evaluation_frequency']

    policy_params = variant['alg_params']



    min_memory_size = policy_params['min_memory_size']
    steps_per_cycle = policy_params['steps_per_cycle']
    train_per_cycle = policy_params['train_per_cycle']
    batch_size = policy_params['batch_size']

    lr_a, lr_c, lr_l = policy_params['lr_a'], policy_params['lr_c'], policy_params['lr_l']
    lr_a_now = lr_a  # learning rate for actor
    lr_c_now = lr_c  # learning rate for critic
    lr_l_now = lr_l  # learning rate for critic

    if 'Fetch' in env_name or 'Hand' in env_name:
        s_dim = env.observation_space.spaces['observation'].shape[0]\
                + env.observation_space.spaces['achieved_goal'].shape[0]+ \
                env.observation_space.spaces['desired_goal'].shape[0]
    else:
        s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    # if disturber_params['process_noise']:
    #     d_dim = disturber_params['noise_dim']
    # else:
    #     d_dim = env_params['disturbance dim']
    a_upperbound = env.action_space.high
    a_lowerbound = env.action_space.low
    policy = SAC_cost(a_dim,s_dim, policy_params)


    pool_params = {
        's_dim': s_dim,
        'a_dim': a_dim,
        'd_dim': 1,
        'store_last_n_paths': store_last_n_paths,
        'memory_capacity': policy_params['memory_capacity'],
        'min_memory_size': policy_params['min_memory_size'],
    }
    if 'value_horizon' in policy_params.keys():
        pool_params.update({'value_horizon': policy_params['value_horizon']})
    else:
        pool_params['value_horizon'] = None
    pool = Pool(pool_params)
    # For analyse
    Render = env_params['eval_render']

    # Training setting
    t1 = time.time()
    global_step = 0
    last_training_paths = deque(maxlen=store_last_n_paths)
    training_started = False

    log_path = variant['log_path']
    logger.configure(dir=log_path, format_strs=['csv'])
    logger.logkv('tau', policy_params['tau'])
    logger.logkv('alpha3', policy_params['alpha3'])
    logger.logkv('batch_size', policy_params['batch_size'])
    logger.logkv('target_entropy', policy.target_entropy)

    for i in range(max_episodes):

        current_path = {'rewards': [],

                        'a_loss': [],

                        'alpha': [],
                        'lambda': [],
                        'lyapunov_error': [],

                        'critic_error': [],
                        'entropy': [],
                        }

        if global_step > max_global_steps:
            break

        s = env.reset()
        if 'Fetch' in env_name or 'Hand' in env_name:
            s = np.concatenate([s[key] for key in s.keys()])

        for j in range(max_ep_steps):
            if Render:
                env.render()
            a = policy.choose_action(s)
            action = a_lowerbound + (a + 1.) * (a_upperbound - a_lowerbound) / 2

            # Run in simulator

            s_, r, done, info = env.step(action)

            if 'Fetch' in env_name or 'Hand' in env_name:
                s_ = np.concatenate([s_[key] for key in s_.keys()])
                if info['done'] > 0:
                    done = True

            if training_started:
                global_step += 1

            if j == max_ep_steps - 1:
                done = True

            terminal = 1. if done else 0.
            pool.store(s, a, np.zeros([1]), np.zeros([1]), r, terminal, s_)
            # policy.store_transition(s, a, disturbance, r,0, terminal, s_)
            # Learn

            if pool.memory_pointer > min_memory_size and global_step % steps_per_cycle == 0:
                training_started = True

                for _ in range(train_per_cycle):
                    batch = pool.sample(batch_size)
                    labda, alpha, c1_loss, c2_loss, l_loss, entropy, a_loss = policy.learn(lr_a_now, lr_c_now, lr_l_now, batch)



            if training_started:
                current_path['rewards'].append(r)
                current_path['lyapunov_error'].append(l_loss)
                current_path['critic_error'].append(min(c1_loss, c2_loss))
                current_path['alpha'].append(alpha)
                current_path['lambda'].append(labda)
                current_path['entropy'].append(entropy)
                current_path['a_loss'].append(a_loss)



            if training_started and global_step % evaluation_frequency == 0 and global_step > 0:

                logger.logkv("total_timesteps", global_step)

                training_diagnotic = evaluate_training_rollouts(last_training_paths)
                if training_diagnotic is not None:
                    if variant['num_of_evaluation_paths'] > 0:
                        eval_diagnotic = training_evaluation(variant, env, policy)
                        [logger.logkv(key, eval_diagnotic[key]) for key in eval_diagnotic.keys()]
                        training_diagnotic.pop('return')
                    [logger.logkv(key, training_diagnotic[key]) for key in training_diagnotic.keys()]
                    logger.logkv('lr_a', lr_a_now)
                    logger.logkv('lr_c', lr_c_now)
                    logger.logkv('lr_l', lr_l_now)

                    string_to_print = ['time_step:', str(global_step), '|']
                    if variant['num_of_evaluation_paths'] > 0:
                        [string_to_print.extend([key, ':', str(eval_diagnotic[key]), '|'])
                         for key in eval_diagnotic.keys()]
                    [string_to_print.extend([key, ':', str(round(training_diagnotic[key], 2)), '|'])
                     for key in training_diagnotic.keys()]
                    print(''.join(string_to_print))

                logger.dumpkvs()
            # 状态更新
            s = s_

            # OUTPUT TRAINING INFORMATION AND LEARNING RATE DECAY
            if done:
                if training_started:
                    last_training_paths.appendleft(current_path)

                frac = 1.0 - (global_step - 1.0) / max_global_steps
                lr_a_now = lr_a * frac  # learning rate for actor
                lr_c_now = lr_c * frac  # learning rate for critic
                lr_l_now = lr_l * frac  # learning rate for critic

                break
    policy.save_result(log_path)

    print('Running time: ', time.time() - t1)
    return



