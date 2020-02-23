
import tensorflow as tf
import numpy as np
import time
import tensorflow_probability as tfp
from collections import OrderedDict, deque
import os
from copy import deepcopy
from .squash_bijector import SquashBijector



SCALE_DIAG_MIN_MAX = (-20, 2)
SCALE_lambda_MIN_MAX = (0, 1)

class Disturber(object):
    def __init__(self,
                 a_dim,
                 s_dim,

                 variant,

                 action_prior = 'uniform',
                 ):



        ###############################  Model parameters  ####################################
        self.memory_capacity = variant['memory_capacity']

        self.batch_size = variant['batch_size']
        self.impulse_instant = np.random.choice(int(250), [1])
        gamma = variant['gamma']
        tau = variant['tau']
        self.energy_decay = variant['energy_decay_rate']
        self.energy_bounded = variant['energy_bounded']
        if self.energy_bounded:
            ita = 0
        else:
            ita = variant['ita']
        self.disturbance_chanel_list=variant['disturbance_chanel_list']
        self.magnitude = variant['disturbance_magnitude'][self.disturbance_chanel_list]
        self.approx_value = True if 'approx_value' not in variant.keys() else variant['approx_value']
        self.memory = np.zeros((self.memory_capacity, s_dim * 2 + a_dim + 3), dtype=np.float32)
        self.pointer = 0

        self.sess = tf.Session()
        self._action_prior = action_prior
        self.a_dim, self.s_dim, = a_dim, s_dim,

        target_entropy = variant['target_entropy']
        if target_entropy is None:
            self.target_entropy = -self.a_dim  #lower bound of the policy entropy
        else:
            self.target_entropy = target_entropy

        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.a_input = tf.placeholder(tf.float32, [None, a_dim], 'a_input')

        self.real_a_input = tf.placeholder(tf.float32, [None, a_dim], 'real_a_input')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.terminal = tf.placeholder(tf.float32, [None, 1], 'terminal')
        self.LR_A = tf.placeholder(tf.float32, None, 'LR_A')
        self.LR_C = tf.placeholder(tf.float32, None, 'LR_C')



        alpha = variant['alpha']

        with tf.variable_scope('disturber'):
            log_alpha = tf.get_variable('alpha', None, tf.float32, initializer=tf.log(alpha))  # Entropy Temperature

            self.alpha = tf.clip_by_value(tf.exp(log_alpha), *SCALE_lambda_MIN_MAX)

            self.a, self.deterministic_a, self.a_dist = self._build_a(self.S, )  # 这个网络用于及时更新参数
            self.q1 = self._build_c(self.S, self.a_input, 'critic1')  # 这个网络是用于及时更新参数
            self.q2 = self._build_c(self.S, self.a_input, 'critic2')  # 这个网络是用于及时更新参数


            self.q1_a = self._build_c(self.S, self.a, 'critic1', reuse=True)
            self.q2_a = self._build_c(self.S, self.a, 'critic2', reuse=True)


            self.adaptive_alpha = variant['adaptive_alpha']

            a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='disturber/Actor')
            c1_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='disturber/critic1')
            c2_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='disturber/critic2')


            ###############################  Model Learning Setting  ####################################
            ema = tf.train.ExponentialMovingAverage(decay=1 - tau)  # soft replacement
            def ema_getter(getter, name, *args, **kwargs):
                return ema.average(getter(name, *args, **kwargs))
            target_update = [ema.apply(a_params), ema.apply(c1_params),ema.apply(c2_params)]  # soft update operation

            # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
            a_, _, a_dist_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)  # replaced target parameters

            # self.cons_a_input_ = tf.placeholder(tf.float32, [None, a_dim, 'cons_a_input_'])
            # self.log_pis = log_pis = self.a_dist.log_prob(self.a)
            self.log_pis = log_pis = self.a_dist.log_prob(self.a)


            # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
            q1_ = self._build_c(self.S_, a_,'critic1', reuse=True, custom_getter=ema_getter)
            q2_ = self._build_c(self.S_, a_, 'critic2', reuse=True, custom_getter=ema_getter)



        alpha_loss = -tf.reduce_mean(log_alpha * tf.stop_gradient(log_pis + self.target_entropy))
        self.alpha_train = tf.train.AdamOptimizer(self.LR_A).minimize(alpha_loss, var_list=log_alpha)

        if self._action_prior == 'normal':
            policy_prior = tf.contrib.distributions.MultivariateNormalDiag(
                loc=tf.zeros(self.a_dim),
                scale_diag=tf.ones(self.a_dim))
            policy_prior_log_probs = policy_prior.log_prob(self.a)
        elif self._action_prior == 'uniform':
            policy_prior_log_probs = 0.0

        min_Q_target = tf.reduce_min((self.q1_a, self.q2_a), axis=0)
        self.a_loss = tf.reduce_mean(self.alpha * tf.expand_dims(log_pis, axis=1) - min_Q_target - policy_prior_log_probs)

        self.atrain = tf.train.AdamOptimizer(self.LR_A).minimize(self.a_loss, var_list=a_params)  #以learning_rate去训练，方向是minimize loss，调整列表参数，用adam

        next_log_pis = a_dist_.log_prob(a_)
        with tf.control_dependencies(target_update):  # soft replacement happened at here
            min_next_q = tf.reduce_min([q1_, q2_], axis=0)
            q1_target = self.R-ita * tf.expand_dims(tf.norm(self.real_a_input, axis=1), axis=1) + gamma * (1 - self.terminal) * tf.stop_gradient(
                min_next_q)  # ddpg
            q2_target = self.R- ita * tf.expand_dims(tf.norm(self.real_a_input, axis=1), axis=1) + gamma * (1 - self.terminal) * tf.stop_gradient(
                min_next_q)  # ddpg


            self.td_error1 = tf.losses.mean_squared_error(labels=q1_target, predictions=self.q1)
            self.td_error2 = tf.losses.mean_squared_error(labels=q2_target, predictions=self.q2)

            self.ctrain1 = tf.train.AdamOptimizer(self.LR_C).minimize(self.td_error1, var_list=c1_params)
            self.ctrain2 = tf.train.AdamOptimizer(self.LR_C).minimize(self.td_error2, var_list=c2_params)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.diagnotics = [ self.alpha, self.td_error1, self.td_error2, tf.reduce_mean(-self.log_pis), self.a_loss]

        self.opt = [self.ctrain1, self.ctrain2, ]
        self.opt.append(self.atrain)
        if self.adaptive_alpha is True:
            self.opt.append(self.alpha_train)

    def choose_action(self, s, k, evaluation=False):
        if evaluation is True:
            return self.sess.run(self.deterministic_a, {self.S: s[np.newaxis, :]})[0]
        else:
            raw_a = self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]
            if self.energy_bounded:
                a = -self.magnitude + (raw_a + 1.) * self.magnitude
                a = a * np.exp(-self.energy_decay * abs(k-self.impulse_instant))
            else:
                a = -self.magnitude + (raw_a + 1.) * self.magnitude
                # a = raw_a

            return a, raw_a

    def learn(self, LR_A, LR_C, batch):
        bs = batch['s']  # state
        ba = batch['raw_d']  # action
        b_real_a = batch['d']
        br = batch['r']  # reward

        bterminal = batch['terminal']
        bs_ = batch['s_']  # next state

        feed_dict = {self.a_input: ba,self.real_a_input: b_real_a, self.S: bs, self.S_: bs_, self.R: br,
                     self.terminal: bterminal, self.LR_C: LR_C, self.LR_A: LR_A}

        self.sess.run(self.opt, feed_dict)
        alpha, q1_error, q2_error,  entropy, a_loss = self.sess.run(self.diagnotics, feed_dict)

        return alpha, q1_error, q2_error, entropy, a_loss

    def store_transition(self, s, a, r, terminal, s_):
        transition = np.hstack((s, a, [r], [l_r], [terminal], s_))
        index = self.pointer % self.memory_capacity  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1


    #action 选择模块也是actor模块


    def _build_a(self, s, name='Actor', reuse=None, custom_getter=None):
        if reuse is None:
            trainable = True
        else:
            trainable = False

        with tf.variable_scope(name, reuse=reuse, custom_getter=custom_getter):
            batch_size = tf.shape(s)[0]

            base_distribution = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(self.a_dim), scale_diag=tf.ones(self.a_dim))
            epsilon = base_distribution.sample(batch_size)
            ## Construct the feedforward action
            net_0 = tf.layers.dense(s, 64, activation=tf.nn.relu, name='l1', trainable=trainable)#原始是30
            net_1 = tf.layers.dense(net_0, 64, activation=tf.nn.relu, name='l4', trainable=trainable)  # 原始是30
            mu = tf.layers.dense(net_1, self.a_dim, activation= None, name='a', trainable=trainable)
            log_sigma = tf.layers.dense(net_1, self.a_dim, None, trainable=trainable)
            log_sigma = tf.clip_by_value(log_sigma, *SCALE_DIAG_MIN_MAX)
            sigma = tf.exp(log_sigma)

            # if not self.energy_bounded:
            if False:
                bijector = tfp.bijectors.Affine(shift=mu, scale_diag=sigma)
                raw_action = bijector.forward(epsilon)


                ## Construct the distribution
                bijector = tfp.bijectors.Chain((

                    tfp.bijectors.Affine(
                        shift=mu,
                        scale_diag=sigma),
                ))
                distribution = tfp.distributions.ConditionalTransformedDistribution(
                        distribution=base_distribution,
                        bijector=bijector)

                return raw_action, mu, distribution
            else:
                squash_bijector = (SquashBijector())
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
            n_l1 = 64#30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net_0 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net_1 = tf.layers.dense(net_0, 64, activation=tf.nn.relu, name='l2', trainable=trainable)  # 原始是30
            return tf.layers.dense(net_1, 1, trainable=trainable)  # Q(s,a)

    def save_result(self, path):

        save_path = self.saver.save(self.sess, path + "/disturber/model.ckpt")
        print("Save to path: ", save_path)

    def restore(self, path):
        model_file = tf.train.latest_checkpoint(path+'disturber/')
        self.saver.restore(self.sess, model_file)




