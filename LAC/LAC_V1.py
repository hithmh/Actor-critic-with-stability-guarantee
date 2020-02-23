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
from robustness_eval import training_evaluation
from disturber.disturber import Disturber
from pool.pool import Pool
import logger
from variant import *

SCALE_DIAG_MIN_MAX = (-20, 2)
SCALE_lambda_MIN_MAX = (0, 1)


class LAC(object):
    def __init__(self,
                 a_dim,
                 s_dim,


                 variant,

                 action_prior = 'uniform',
                 ):



        ###############################  Model parameters  ####################################
        # self.memory_capacity = variant['memory_capacity']

        self.batch_size = variant['batch_size']
        gamma = variant['gamma']

        tau = variant['tau']
        self.approx_value = True if 'approx_value' not in variant.keys() else variant['approx_value']
        # self.memory = np.zeros((self.memory_capacity, s_dim * 2 + a_dim+ d_dim + 3), dtype=np.float32)
        # self.pointer = 0
        self.sess = tf.Session()
        self._action_prior = action_prior
        s_dim = s_dim * (variant['history_horizon']+1)
        self.a_dim, self.s_dim, = a_dim, s_dim
        self.history_horizon = variant['history_horizon']
        self.working_memory = deque(maxlen=variant['history_horizon']+1)
        target_entropy = variant['target_entropy']
        if target_entropy is None:
            self.target_entropy = -self.a_dim   #lower bound of the policy entropy
        else:
            self.target_entropy = target_entropy
        self.form_of_lyapunov = variant['form_of_lyapunov']

        with tf.variable_scope('Actor'):
            self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
            self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
            self.a_input = tf.placeholder(tf.float32, [None, a_dim], 'a_input')
            self.a_input_ = tf.placeholder(tf.float32, [None, a_dim], 'a_input_')
            self.R = tf.placeholder(tf.float32, [None, 1], 'r')
            self.R_N_ = tf.placeholder(tf.float32, [None, 1], 'r_N_')
            self.V = tf.placeholder(tf.float32, [None, 1], 'v')
            self.terminal = tf.placeholder(tf.float32, [None, 1], 'terminal')
            self.LR_A = tf.placeholder(tf.float32, None, 'LR_A')
            self.LR_lag = tf.placeholder(tf.float32, None, 'LR_lag')
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

            self.l = self._build_l(self.S, self.a_input)   # lyapunov 网络


            self.use_lyapunov = variant['use_lyapunov']
            self.adaptive_alpha = variant['adaptive_alpha']

            a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor/actor')
            l_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor/Lyapunov')

            ###############################  Model Learning Setting  ####################################
            ema = tf.train.ExponentialMovingAverage(decay=1 - tau)  # soft replacement

            def ema_getter(getter, name, *args, **kwargs):
                return ema.average(getter(name, *args, **kwargs))
            target_update = [ema.apply(a_params),  ema.apply(l_params)]  # soft update operation

            # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
            a_, _, a_dist_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)  # replaced target parameters
            lya_a_, _, lya_a_dist_ = self._build_a(self.S_, reuse=True)
            # self.cons_a_input_ = tf.placeholder(tf.float32, [None, a_dim, 'cons_a_input_'])
            # self.log_pis = log_pis = self.a_dist.log_prob(self.a)
            self.log_pis = log_pis = self.a_dist.log_prob(self.a)
            self.prob = tf.reduce_mean(self.a_dist.prob(self.a))

            # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
            # self.l_actor = self._build_l(self.S, self.a, reuse=True)
            l_ = self._build_l(self.S_, a_, reuse=True, custom_getter=ema_getter)
            self.l_ = self._build_l(self.S_, lya_a_, reuse=True)

            # lyapunov constraint
            self.l_derta = tf.reduce_mean(self.l_ - self.l + (alpha3) * self.R)
            # self.l_derta = tf.reduce_mean(self.l_ - self.l_actor + (alpha3) * self.R)

            labda_loss = -tf.reduce_mean(log_labda * self.l_derta)
            alpha_loss = -tf.reduce_mean(log_alpha * tf.stop_gradient(log_pis + self.target_entropy))
            self.alpha_train = tf.train.AdamOptimizer(self.LR_A).minimize(alpha_loss, var_list=log_alpha)
            self.lambda_train = tf.train.AdamOptimizer(self.LR_lag).minimize(labda_loss, var_list=log_labda)

            if self._action_prior == 'normal':
                policy_prior = tf.contrib.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(self.a_dim),
                    scale_diag=tf.ones(self.a_dim))
                policy_prior_log_probs = policy_prior.log_prob(self.a)
            elif self._action_prior == 'uniform':
                policy_prior_log_probs = 0.0

            if self.use_lyapunov is True:
                a_loss = self.labda * self.l_derta + self.alpha * tf.reduce_mean(log_pis) - policy_prior_log_probs
            else:
                a_loss = a_preloss

            self.a_loss = a_loss
            self.atrain = tf.train.AdamOptimizer(self.LR_A).minimize(a_loss, var_list=a_params)

            next_log_pis = a_dist_.log_prob(a_)
            with tf.control_dependencies(target_update):  # soft replacement happened at here
                if self.form_of_lyapunov == 'inf':
                    l_target = self.R + gamma * (1 - self.terminal) * tf.stop_gradient(l_)  # Lyapunov critic - self.alpha * next_log_pis
                elif self.form_of_lyapunov == 'finite':
                    l_target = self.V
                elif self.form_of_lyapunov == 'entire_horizon':
                    l_target = self.V
                elif self.form_of_lyapunov == 'soft_horizon':
                    l_target = self.R - self.R_N_ + tf.stop_gradient(l_)
                elif self.form_of_lyapunov == 'cost':
                    l_target = self.R
                else:
                    raise IndexError
                    print('unrecogonized Lyapunov form')


                self.l_error = tf.losses.mean_squared_error(labels=l_target, predictions=self.l)
                self.ltrain = tf.train.AdamOptimizer(self.LR_L).minimize(self.l_error, var_list=l_params)

            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            self.diagnotics = [self.labda, self.alpha, self.l_error, tf.reduce_mean(-self.log_pis), self.a_loss]

            if self.use_lyapunov is True:
                self.opt = [self.ltrain, self.lambda_train]
            self.opt.append(self.atrain)
            if self.adaptive_alpha is True:
                self.opt.append(self.alpha_train)

    def choose_action(self, s, evaluation = False):
        if len(self.working_memory) < self.history_horizon:
            [self.working_memory.appendleft(s) for _ in range(self.history_horizon)]

        self.working_memory.appendleft(s)
        try:
            s = np.concatenate(self.working_memory)
        except ValueError:
            print(s)

        if evaluation is True:
            try:
                return self.sess.run(self.deterministic_a, {self.S: s[np.newaxis, :]})[0]
            except ValueError:
                return
        else:
            return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self, LR_A, LR_C, LR_L,LR_lag, batch):

        bs = batch['s']  # state
        ba = batch['a']  # action

        br = batch['r']  # reward

        bterminal = batch['terminal']
        bs_ = batch['s_']  # next state
        feed_dict = {self.a_input: ba,  self.S: bs, self.S_: bs_, self.R: br, self.terminal: bterminal,
                     self.LR_C: LR_C, self.LR_A: LR_A, self.LR_L: LR_L, self.LR_lag:LR_lag}

        bv = batch['value']
        b_r_ = batch['r_N_']
        feed_dict.update({self.V:bv, self.R_N_:b_r_})

        self.sess.run(self.opt, feed_dict)
        labda, alpha, l_error, entropy, a_loss = self.sess.run(self.diagnotics, feed_dict)

        return labda, alpha, l_error, entropy, a_loss

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


    def evaluate_value(self, s, a):

        if len(self.working_memory) < self.history_horizon:
            [self.working_memory.appendleft(s) for _ in range(self.history_horizon)]

        self.working_memory.appendleft(s)
        try:
            s = np.concatenate(self.working_memory)
        except ValueError:
            print(s)

        return self.sess.run(self.l, {self.S: s[np.newaxis, :], self.a_input: a[np.newaxis, :]})[0]

    def _build_l(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Lyapunov', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 64#30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net_0 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net_1 = tf.layers.dense(net_0, 64, activation=tf.nn.relu, name='l2', trainable=trainable)  # 原始是30
            # return tf.abs(tf.layers.dense(net_1, 1, trainable=trainable))  # Q(s,a)
            net_2 = tf.layers.dense(net_1, 16, activation=tf.nn.relu, name='l3', trainable=trainable)  # 原始是30
            return tf.expand_dims(tf.reduce_sum(tf.square(net_2), axis=1),axis=1)  # Q(s,a)
            # return tf.square(tf.layers.dense(net_1, 1, trainable=trainable)) # Q(s,a)

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
    store_last_n_paths = variant['store_last_n_paths']
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
    policy = LAC(a_dim,s_dim, policy_params)

    pool_params = {
        's_dim': s_dim,
        'a_dim': a_dim,
        'd_dim': 1,
        'store_last_n_paths': store_last_n_paths,

        'memory_capacity': policy_params['memory_capacity'],
        'min_memory_size': policy_params['min_memory_size'],
        'history_horizon': policy_params['history_horizon'],
        'value_horizon': policy_params['value_horizon'],
        'form_of_lyapunov': policy_params['form_of_lyapunov']
    }
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
            # a = a*0
            action = a_lowerbound + (a + 1.) * (a_upperbound - a_lowerbound) / 2

            # Run in simulator
            disturbance_input = np.zeros([a_dim + s_dim])

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

            if pool.memory_pointer > min_memory_size and global_step % steps_per_cycle == 0:
                training_started = True

                for _ in range(train_per_cycle):
                    batch = pool.sample(batch_size)
                    labda, alpha, l_loss, entropy, a_loss = policy.learn(lr_a_now, lr_c_now, lr_l_now, lr_a, batch)

            if training_started:
                current_path['rewards'].append(r)
                current_path['lyapunov_error'].append(l_loss)
                current_path['alpha'].append(alpha)
                current_path['lambda'].append(labda)
                current_path['entropy'].append(entropy)
                current_path['a_loss'].append(a_loss)



            if training_started and global_step % evaluation_frequency == 0 and global_step > 0:

                logger.logkv("total_timesteps", global_step)

                training_diagnotic = evaluate_training_rollouts(last_training_paths)
                if training_diagnotic is not None:
                    eval_diagnotic = training_evaluation(variant, env, policy)
                    [logger.logkv(key, eval_diagnotic[key]) for key in eval_diagnotic.keys()]
                    training_diagnotic.pop('return')
                    [logger.logkv(key, training_diagnotic[key]) for key in training_diagnotic.keys()]
                    logger.logkv('lr_a', lr_a_now)
                    logger.logkv('lr_c', lr_c_now)
                    logger.logkv('lr_l', lr_l_now)

                    string_to_print = ['time_step:', str(global_step), '|']
                    [string_to_print.extend([key, ':', str(eval_diagnotic[key]), '|'])
                     for key in eval_diagnotic.keys()]
                    [string_to_print.extend([key, ':', str(round(training_diagnotic[key], 2)) , '|'])
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


