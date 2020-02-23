
import tensorflow as tf
import numpy as np
import time
from .squash_bijector import SquashBijector
import tensorflow_probability as tfp
from collections import OrderedDict, deque
import os
from copy import deepcopy
from baselines.variant import VARIANT, get_env_from_name, get_policy, get_train
from .utils import get_evaluation_rollouts, evaluate_rollouts, evaluate_training_rollouts
from baselines import logger
from baselines.safety_constraints import get_safety_constraint_func

SCALE_DIAG_MIN_MAX = (-20, 2)
SCALE_lambda_MIN_MAX = (0, 1)



class Dreamer(object):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.LR_D= tf.placeholder(tf.float32, None, 'LR_D')
        self.LR_R = tf.placeholder(tf.float32, None, 'LR_R')

        self.x_threshold = 5
        self.A = tf.placeholder(tf.float32, [None, a_dim], 'a')
        self.dreamer = self._build_dreamer(self.S, self.A, )
        self.score=self._build_score(self.dreamer)

        d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Dreamer')
        r_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Score')

        self.viewer = None
        self.state = None


        self.dreamer_loss_s = tf.reduce_mean(tf.squared_difference(self.S_, self.dreamer))

        self.dreamer_loss_r = tf.reduce_mean(tf.squared_difference(self.R, self.score))

        self.dreamertrain_s = tf.train.AdamOptimizer(self.LR_D).minimize(self.dreamer_loss_s,var_list = d_params)

        self.dreamertrain_r = tf.train.AdamOptimizer(self.LR_R).minimize(self.dreamer_loss_r, var_list=r_params)


        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, "Model/SRDDPG_D_V1.ckpt")  # 1 0.1 0.5 0.001
    def dream(self, s,a):
        self.state=self.sess.run(self.dreamer, {self.S: s[np.newaxis, :],self.A: a[np.newaxis, :]})[0]
        return self.sess.run(self.dreamer, {self.S: s[np.newaxis, :],self.A: a[np.newaxis, :]})[0],self.sess.run(self.score, {self.S: s[np.newaxis, :],self.A: a[np.newaxis, :]})[0]

    def learn(self,LR_D,LR_R):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        bs_ = bt[:, -self.s_dim:]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        self.sess.run(self.dreamertrain_s, {self.S: bs,self.A: ba, self.S_: bs_, self.LR_D: LR_D})
        self.sess.run(self.dreamertrain_r, {self.S: bs, self.A: ba, self.R: br, self.LR_R: LR_R})

        return self.sess.run(self.dreamer_loss_s, {self.S: bs,self.A: ba, self.S_: bs_}),self.sess.run(self.dreamer_loss_r, {self.S: bs,self.A: ba, self.R: br})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_dreamer(self, s, a, reuse=None, custom_getter=None):

        if reuse is None:
            trainable = True
        else:
            trainable = False

        with tf.variable_scope(name, reuse=reuse, custom_getter=custom_getter):
            batch_size = tf.shape(s)[0]
            squash_bijector = (SquashBijector())
            base_distribution = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(self.s_dim),
                                                                         scale_diag=tf.ones(self.s_dim))
            epsilon = base_distribution.sample(batch_size)
            ## Construct the feedforward action
            net_0 = tf.layers.dense(s, 256, activation=tf.nn.relu, name='l1', trainable=trainable)  # 原始是30
            net_1 = tf.layers.dense(net_0, 256, activation=tf.nn.relu, name='l4', trainable=trainable)  # 原始是30
            mu = tf.layers.dense(net_1, self.a_dim, activation=None, name='a', trainable=trainable)
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

    def _build_score(self, s,reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Score', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 512  # 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net_0 = tf.nn.relu(tf.matmul(s, w1_s)+ b1)
            net_1 = tf.layers.dense(net_0, 512, activation=tf.nn.relu, name='l2', trainable=trainable)
            net_2 = tf.layers.dense(net_1, 256, activation=tf.nn.relu, name='l3', trainable=trainable)
            return tf.layers.dense(net_2, 1, trainable=trainable)

    def save_result(self):
        save_path = self.saver.save(self.sess, "Model/SRDDPG_D_ONLINE.ckpt")
        print("Save to path: ", save_path)


    def restore(self, path):
        model_file = tf.train.latest_checkpoint(path+'/')
        self.saver.restore(self.sess,model_file)


