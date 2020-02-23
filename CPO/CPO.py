"""
A simple version of Proximal Policy Optimization (PPO) using single thread.
Based on:
1. Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]
2. Proximal Policy Optimization Algorithms (OpenAI): [https://arxiv.org/abs/1707.06347]
View more on my tutorial website: https://morvanzhou.github.io/tutorials
Dependencies:
tensorflow r1.2
gym 0.9.2
"""

import tensorflow as tf
# import tensorflow_probability as tfp
import numpy as np
from collections import OrderedDict, deque
import logger
from copy import deepcopy
import os
import time
from variant import VARIANT, get_env_from_name, get_policy, get_train
from .utils import get_evaluation_rollouts, evaluate_rollouts, evaluate_training_rollouts
from robustness_eval import training_evaluation
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

SCALE_DIAG_MIN_MAX = (-20, 0)
def new_tensor(name, ndim, dtype):

    return tf.placeholder(dtype, ndim, name=name)


def new_tensor_like(name, arr_like):
    return new_tensor(name, arr_like.shape, arr_like.dtype)

def get_hessians_for_flatten_variables(Heassians, var_shapes):
    flattened_H = []
    for h, var_shape in zip(Heassians,var_shapes):
        flattened_h = []
        if len(var_shape) >1:
            for i in range(var_shape[0]):
                for j in range(var_shape[1]):
                    flattened_h.append(tf.reshape(h[i,j], [-1, var_shape[0]*var_shape[1]]))
            flattened_h = tf.concat(flattened_h, axis=0)
            # flattened_h = tf.reshape(h, [-1, var_shape[0] * var_shape[1]])
        else:
            flattened_h = h
        flattened_H.append(flattened_h)
    return flattened_H

def flatten_tensors(tensors):
    if len(tensors) > 0:
        return np.concatenate([np.reshape(x, [-1]) for x in tensors])
    else:
        return np.asarray([])

def flatten_tf_tensors(tensors):
    if len(tensors) > 0:
        return tf.concat([tf.reshape(x, [-1]) for x in tensors],0)
    else:
        return []




def split_flatten(flattened, tensor_shapes):
    tensor_sizes = list(map(np.prod, tensor_shapes))
    indices = np.cumsum(tensor_sizes)[:-1]
    return np.split(flattened, indices)

def cg(f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
    """
    Demmel p 312
    """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)

    rdotr = r.dot(r)

    fmtstr = "%10i %10.3g %10.3g"
    titlestr = "%10s %10s %10s"
    if verbose: print(titlestr % ("iter", "residual norm", "soln norm"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
        if verbose: print(fmtstr % (i, rdotr, np.linalg.norm(x)))
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    if callback is not None:
        callback(x)
    if verbose: print(fmtstr % (i + 1, rdotr, np.linalg.norm(x)))  # pylint: disable=W0631
    return x


def f_Ax(A,x,shapes):
    split_x = split_flatten(x, shapes)

    return np.concatenate([np.matmul(h,x) for h,x in zip(A, split_x)])

def cg_try(A, b, shapes, cg_iters=10, callback=None, verbose=True, residual_tol=1e-10):
    """
    Demmel p 312
    """

    # p = b.copy()
    # r = b.copy()
    x = np.zeros_like(b)
    r = b - f_Ax(A,x, shapes)
    p = r.copy()
    rdotr = r.dot(r)

    fmtstr = "%10i %10.3g %10.3g"
    titlestr = "%10s %10s %10s"
    if verbose: print(titlestr % ("iter", "residual norm", "soln norm"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
        if verbose: print(fmtstr % (i, rdotr, np.linalg.norm(x)))
        z = f_Ax(A,p, shapes)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    if callback is not None:
        callback(x)
    if verbose: print(fmtstr % (i + 1, rdotr, np.linalg.norm(x)))  # pylint: disable=W0631
    return x



class Heassian_for_flat_params(object):

    def __init__(self, f, target, reg_coeff, sess):
        self.reg_coeff = reg_coeff
        params = target.get_params()

        self.target = target
        self.sess = sess
        gradients = tf.gradients(f, params)
        # flattened_grads = flatten_tf_tensors(gradients)

        H = tf.hessians(f,params[0])
        self.H = H
        self.xs = new_tensor_like(None, params[0])
        Hx_plain_splits = tf.gradients(tf.reduce_sum(gradients[0] * self.xs), params[0])
        self.f_Hx_plain = Hx_plain_splits[0]

        # Hx_plain_splits = [tf.matmul(H, x[:,tf.newaxis]) for H,x in zip(self.H, self.xs)]
        # self.f_Hx_plain = flatten_tf_tensors(Hx_plain_splits)


    def build_eval(self, feed_dict):

        def eval(x):
            # np.linalg.cholesky(self.sess.run(self.H[0], feed_dict))

            feed_dict.update({self.xs: x})
            return self.sess.run(self.f_Hx_plain + self.reg_coeff * self.xs, feed_dict) # + self.reg_coeff * self.xs[0]

        return eval

    def build_heassians_eval(self, feed_dict):
        self.H_values = self.sess.run(self.H, feed_dict)
        np.linalg.cholesky(self.H_values[0])
        shapes = self.target.get_params_shape()
        def eval(x):

            return f_Ax(self.H_values, x, shapes)

        return eval

class GuassianMLP_from_flatten_params(object):
    def __init__(self, s, a_dim, name, trainable, shape=[64,32],):

        self.a_dim = a_dim

        self.s_dim = s.shape.as_list()[1]
        shape = [self.s_dim, shape[0], shape[1], self.a_dim]

        def build_layer_with_flat_param(input, params, shape, name, activation = None):

            flat_w = params[0]
            flat_b = params[1]
            with tf.name_scope(name):
                w = tf.reshape(flat_w, [shape[0], shape[1]])
                b = tf.reshape(flat_b, [1, shape[1]])
                if activation:
                    output = activation(tf.matmul(input, w) + b)
                else:
                    output =tf.matmul(input,w) + b
            return output

        with tf.variable_scope(name):
            total_size = 0
            shape_list = []
            reshape_list = []
            for i in range(len(shape)):
                if i == len(shape)-1:
                    total_size += shape[i-1] * shape[i] + shape[i]
                    shape_list.extend([shape[i-1] * shape[i], shape[i]])
                    reshape_list.append([shape[i-1], shape[i]])
                else:
                    total_size += shape[i] * shape[i + 1] + shape[i + 1]
                    shape_list.extend([shape[i] * shape[i + 1], shape[i + 1]])
                    reshape_list.append([shape[i], shape[i+1]])

            flat_params = tf.get_variable(name + '_param', [total_size], trainable=trainable)
            split_params = tf.split(flat_params, shape_list)
            net = s
            for i in range(len(shape)-2):
                net = build_layer_with_flat_param(net,split_params[2*i:2*i+2], reshape_list[i],'l'+str(i),tf.nn.tanh)
            i += 1
            self.mu = build_layer_with_flat_param(net, split_params[2*i:2*i+2], reshape_list[i], 'mu')

            i += 1
            self.log_sigma= build_layer_with_flat_param(net, split_params[2*i:2*i+2], reshape_list[i], 'log_sigma')
            # self.log_sigma = tf.clip_by_value(self.log_sigma, *SCALE_DIAG_MIN_MAX)

            # base_distribution = tfp.distributions.MultivariateNormalDiag(loc=self.mu, scale_diag=tf.exp(self.log_sigma))
            base_distribution = tf.contrib.distributions.MultivariateNormalDiag(loc=self.mu, scale_diag=tf.exp(self.log_sigma))
            self.norm_dist = base_distribution
        self.params = [flat_params]

        self.name = name
        self.params_shape = [param.shape.as_list() for param in (self.params)]
    def get_params(self):
        return self.params

    def get_params_values(self, sess):
        param_values = sess.run(self.params)[0]

        return param_values

    def set_params(self, flattened_params, sess):
        new_params = flattened_params
        set_params_op = self.params[0].assign(new_params)
        sess.run(set_params_op)

    # def get_params_shape(self):
    #
    #     return self.params_shape

    def entropy(self):
        return tf.reduce_mean(self.norm_dist.entropy())
        # return tf.reduce_mean(tf.reduce_sum(self.log_sigma + .5 * np.log(2.0 * np.pi * np.e), axis=-1))

    # def flat_to_params(self, flattened_params):
    #     return unflatten_tensors(flattened_params, self.get_params_shape())

class SPPO(object):
    def __init__(self, a_dim, s_dim, args,
                 delta=0.01,
                 reg_coeff = 1e-5,
                 cg_iters=10,
                 C_UPDATE_STEPS=10,
                 L_UPDATE_STEPS=10,
                 backtrack_ratio=0.8,
                 max_backtracks=15,
                 subsample_factor=0.2,
                 attempt_feasible_recovery=True,
                 attempt_infeasible_recovery=True,
                 revert_to_last_safe_point=False,
                 accept_violation=False,
                 verbose_cg=False,
                 linesearch_infeasible_recovery=True,

                 init_beta = 1.,

                 ):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.a_dim, self.s_dim = a_dim, s_dim,

        self.d_0 = args['d_0']
        self.beta = init_beta
        self.N = args['number_of_trajectory']
        # Optimizer settings
        self.attempt_feasible_recovery = attempt_feasible_recovery
        self.attempt_infeasible_recovery = attempt_infeasible_recovery
        self.revert_to_last_safe_point = revert_to_last_safe_point
        self._accept_violation = accept_violation
        self._constraint_name_1 = 'trust region constraint'
        self._constraint_name_2 = 'safety constraint'
        self._subsample_factor = subsample_factor
        self.reg_coeff = args['reg_coeff'] if 'reg_coeff' in args.keys() else reg_coeff
        self._max_quad_constraint_val = args['delta'] if 'delta' in args.keys() else delta
        self._cg_iters = args['cg_iters'] if 'cg_iters' in args.keys() else cg_iters
        self._backtrack_ratio = backtrack_ratio
        self._max_backtracks = max_backtracks
        self.C_UPDATE_STEPS = C_UPDATE_STEPS

        self._verbose_cg = verbose_cg
        self._linesearch_infeasible_recovery = linesearch_infeasible_recovery
        self.expected_safety_threshold = args['safety_threshold']

        self.tfs = tf.placeholder(tf.float32, [None, self.s_dim], 'state')
        # self.cons_S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.tfdc_r = tf.placeholder(tf.float32, [None], 'discounted_r')
        # self.target_of_l = tf.placeholder(tf.float32, [None], 'target_of_l')
        # self.l_r = tf.placeholder(tf.float32, [None], 'l_r')
        self.init_value = tf.placeholder(tf.float32, [], 'D')
        self.rescale = tf.placeholder(tf.float32, [None], 'T')
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None], 'old_v')
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])
        self.LR_C = tf.placeholder(tf.float32, [], 'LR_C')
        self.LR_A = args['lr_a']
        self.t = tf.placeholder(tf.float32, [None,1], 't')
        self.gamma = tf.constant(args['gamma'])
        # self.l = self._build_l(self.tfs)
        self.LR_L = tf.placeholder(tf.float32, [], 'LR_L')
        self.labda = tf.placeholder(tf.float32, None, 'Labda')
        self.tfa = tf.placeholder(tf.float32, [None, self.a_dim], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        self.constraint_adv = tf.placeholder(tf.float32, [None], 'cons_advantage')
        # l_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Lyapunov')

        # ACTOR

        # model_policy = GuassianMLP(self.tfs, self.a_dim, 'model_policy', trainable=True)
        #
        # self.policy = GuassianMLP_from_model(self.tfs, model_policy, 'pi', trainable=True)
        self.policy = GuassianMLP_from_flatten_params(self.tfs, self.a_dim, 'pi', trainable=True)
        self.v = self._build_c(self.tfs, trainable=True)
        self.pi = pi = self.policy.norm_dist
        pi_params = self.policy.get_params()
        self.pi_params = pi_params
        self.entropy = self.policy.entropy()

        self.old_policy = GuassianMLP_from_flatten_params(self.tfs, self.a_dim, 'oldpi', trainable=False)
        self.oldpi = oldpi = self.old_policy.norm_dist
        oldpi_params = self.old_policy.get_params()

        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]
        with tf.name_scope('kl_divergence'):
            # kl = pi.kl_divergence(oldpi)
            self.kl_mean = tf.reduce_mean(tf.contrib.distributions.kl_divergence(pi, oldpi))



        self.epsilon = (1-self.gamma)*(self.d_0-tf.reduce_mean(self.init_value))

        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                ratio = tf.expand_dims(pi.prob(self.tfa) / oldpi.prob(self.tfa), 1)

                self.constraint_func = tf.reduce_sum(ratio * tf.pow(self.gamma, self.t) * (self.tfadv + self.v))/self.N



        f_grads = tf.gradients(self.constraint_func, pi_params)
        self.f_grads = f_grads

        self.Heassian_obj = Heassian_for_flat_params(self.kl_mean, self.policy, self.reg_coeff, self.sess)


        # CRITIC

        # self.closs = tf.reduce_mean(tf.square(self.advantage))
        OLDVPRED = tf.expand_dims(OLDVPRED, axis=1)
        tfdc_r = tf.expand_dims(self.tfdc_r, axis=1)
        vpredclipped = OLDVPRED + tf.clip_by_value(self.v - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        # Unclipped value
        vf_losses1 = tf.losses.mean_squared_error(labels=tfdc_r, predictions=self.v)
        # Clipped value
        vf_losses2 = tf.losses.mean_squared_error(labels=tfdc_r, predictions=vpredclipped)
        # self.closs = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        self.closs = vf_losses1
        self.ctrain = tf.train.AdamOptimizer(self.LR_C).minimize(self.closs)


        self.diagnosis_names = ['value_loss', 'policy_entropy']
        self.diagnosis = [self.closs,  self.entropy]
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        # self.saver.restore(self.sess, "Model/PPO_Lyapunov_V3.ckpt")  # 1 0.1 0.5 0.001


        # optimizer initialization
        self.last_safe_point = None
        self._last_lin_pred_S = 0
        self._last_surr_pred_S = 0

        writer = tf.summary.FileWriter("./cpo_log", self.sess.graph)
        writer.close()
    def choose_action(self, s, evaluation=False):
        if evaluation:
            s = np.squeeze(s)
        s = s[np.newaxis, :]
        a, v = self.sess.run([self.sample_op, self.v], {self.tfs: s})
        if evaluation:
            action = np.tanh(a[0])
            return action
        else:
            return a, v[0]

    def predict_values(self, s):

        return np.squeeze(self.sess.run([self.v], {self.tfs: [s]}))

    def update(self, s, s_, returns,  advs,  a, old_values, time, initial_return, cliprangenow, LR_C, length_of_trajectory):

        # advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        advs = advs[:, np.newaxis]
        time = time[:, np.newaxis]
        feed_dict = {self.tfs: s, self.tfa: a, self.tfadv: advs, self.tfdc_r:returns, self.LR_C: LR_C, self.t:time,
                     self.init_value:initial_return, self.CLIPRANGE:cliprangenow,
                     self.S_:s_, self.OLDVPRED : old_values, self.rescale:[length_of_trajectory,],
                     }


        self.sess.run(self.update_oldpi_op)


        self.optimize(feed_dict)

        # update critic
        [self.sess.run(self.ctrain, feed_dict) for _ in range(self.C_UPDATE_STEPS)]

        return self.sess.run(self.diagnosis, feed_dict)


    def optimize(self, feed_dict):

        flat_g = self.sess.run(self.f_grads, feed_dict)[0]
        epsilon = self.sess.run(self.epsilon, feed_dict)
        # Hx = self.Heassian_obj.build_eval(sub_feed_dict)
        Hx = self.Heassian_obj.build_eval(feed_dict)

        norm_g = np.sqrt(flat_g.dot(flat_g))
        unit_g = flat_g / norm_g
        v = norm_g * cg(Hx, unit_g, cg_iters=self._cg_iters, verbose=self._verbose_cg)
        approx_g = Hx(v)
        # approx_g = f_Ax(H, v, params_shape)
        q = v.dot(approx_g)  # approx = g^T H^{-1} g
        lamda = -epsilon*self.beta/q
        residual = np.sqrt((approx_g - flat_g).dot(approx_g - flat_g))
        prev_param = np.copy(self.policy.get_params_values(self.sess))
        cur_step = lamda * v / self.beta
        cur_param = prev_param - self.LR_A * cur_step
        self.policy.set_params(cur_param, self.sess)

        quad_constraint_val, lin_constraint_val = self.sess.run([self.kl_mean, self.constraint_func], feed_dict)
        if np.isnan(quad_constraint_val) or np.isnan(lin_constraint_val):
            logger.log("Something is NaN. Rejecting the step!")
            if np.isnan(quad_constraint_val):
                logger.log("Violated because quad_constraint %s is NaN" %
                           self._constraint_name_1)
            if np.isnan(lin_constraint_val):
                logger.log("Violated because lin_constraint %s is NaN" %
                           self._constraint_name_2)
            self.policy.set_params(prev_param, self.sess)


    #critic模块
    def _build_c(self, s,trainable):
        with tf.variable_scope('Critic'):
            net_0 = tf.layers.dense(s, 256, activation=tf.nn.relu, name='l0', trainable=trainable)
            net_1 = tf.layers.dense(net_0, 128, activation=tf.nn.relu, name='l1', trainable=trainable)
            # net_2 = tf.layers.dense(net_1, 256, activation=tf.nn.relu, name='l2', trainable=trainable)
            # net_3 = tf.layers.dense(net_2, 128, activation=tf.nn.relu, name='l3', trainable=trainable)
            return tf.layers.dense(net_1, 1, trainable=trainable)  # V(s)

    def _build_q(self, s, a, name ='Q', reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope(name, reuse=reuse, custom_getter=custom_getter):
            n_l1 = 256#30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net_0 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net_1 = tf.layers.dense(net_0, 256, activation=tf.nn.relu, name='l2', trainable=trainable)  # 原始是30
            return tf.layers.dense(net_1, 1, trainable=trainable)  # Q(s,a)
    #Lyapunov
    def _build_l(self, s,reuse=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Lyapunov', reuse=reuse):
            net_0 = tf.layers.dense(s, 256, activation=tf.nn.relu, name='l0', trainable=trainable)
            net_1 = tf.layers.dense(net_0, 256, activation=tf.nn.relu, name='l1', trainable=trainable)
            # net_2 = tf.layers.dense(net_1, 256, activation=tf.nn.relu, name='l2', trainable=trainable)
            # net_3 = tf.layers.dense(net_2, 128, activation=tf.nn.relu, name='l3', trainable=trainable)
            return tf.layers.dense(net_1, 1, trainable=trainable)  # V(s)


    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

    def get_l(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.l, {self.tfs: s})[0, 0]

    def save_result(self):
        save_path = self.saver.save(self.sess, "Model/PPO_Lyapunov_V4.ckpt")
        print("Save to path: ", save_path)

def train(variant):
    env_name = variant['env_name']
    env = get_env_from_name(env_name)

    evaluation_env = get_env_from_name(env_name)
    env_params = variant['env_params']

    max_episodes = env_params['max_episodes']
    max_ep_steps = env_params['max_ep_steps']
    max_global_steps = env_params['max_global_steps']
    store_last_n_paths = variant['store_last_n_paths']
    evaluation_frequency = variant['evaluation_frequency']


    alg_name = variant['algorithm_name']
    policy_build_fn = get_policy(alg_name)
    policy_params = variant['alg_params']
    batch_size = policy_params['batch_size']



    lr_c = policy_params['lr_c']
    cliprange = policy_params['cliprange']
    cliprangenow = cliprange
    lr_c_now = lr_c  # learning rate for critic

    gamma = policy_params['gamma']
    gae_lamda = policy_params['gae_lamda']

    log_path = variant['log_path']
    logger.configure(dir=log_path, format_strs=policy_params['output_format'])
    logger.logkv('safety_threshold', policy_params['safety_threshold'])
    logger.logkv('alpha3', policy_params['alpha3'])
    logger.logkv('batch_size', batch_size)
    if 'Fetch' in env_name or 'Hand' in env_name:
        s_dim = env.observation_space.spaces['observation'].shape[0]\
                + env.observation_space.spaces['achieved_goal'].shape[0]+ \
                env.observation_space.spaces['desired_goal'].shape[0]
    else:
        s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_upperbound = env.action_space.high
    a_lowerbound = env.action_space.low
    policy = policy_build_fn(a_dim, s_dim, policy_params)

    # For analyse
    Render = env_params['eval_render']


    # Training setting
    t1 = time.time()
    global_step = 0
    last_training_paths = deque(maxlen=policy.N)




    for j in range(max_global_steps ):

        if global_step > max_global_steps:
            break

        mb_obs, mb_obs_, mb_rewards,  mb_actions, mb_values, mb_terminals, mb_t = [], [], [], [], [], [], []

        for n in range(policy.N):
            current_path = {'rewards': [],
                            'obs': [],
                            'obs_': [],
                            'done': [],
                            'value': [],
                            't': [],
                            'action': [],
                            }
            s = env.reset()
            if 'Fetch' in env_name or 'Hand' in env_name:
                s = np.concatenate([s[key] for key in s.keys()])
        # For n in range number of steps
            for t in range(max_ep_steps):

                # Given observations, get action value and neglopacs
                # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init

                [a], [value] = policy.choose_action(s)

                action = np.tanh(a)
                action = a_lowerbound + (action + 1.) * (a_upperbound - a_lowerbound) / 2
                # Run in simulator
                s_, r, done, info = env.step(action)
                if 'Fetch' in env_name or 'Hand' in env_name:
                    s_ = np.concatenate([s_[key] for key in s_.keys()])
                if t == max_ep_steps - 1:
                    done = True
                terminal = 1. if done else 0.




                if Render:
                    env.render()

                current_path['rewards'].append(r)
                current_path['action'].append(a)
                current_path['obs'].append(s)
                current_path['obs_'].append(s_)
                current_path['done'].append(terminal)
                current_path['value'].append(value)
                current_path['t'].append(t)
                if done:

                    global_step += t+1
                    last_training_paths.appendleft(current_path)

                    break
                else:
                    s = s_
        # mb_obs = np.asarray(mb_obs, dtype=s.dtype)
        # mb_values = np.asarray(mb_values, dtype=s.dtype)
        # mb_l_values = np.asarray(mb_l_values, dtype=s.dtype)
        # mb_actions = np.asarray(mb_actions, dtype=action.dtype)
        # mb_obs_ = np.asarray(mb_obs_, dtype=s_.dtype)
        # mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        # mb_l_rewards = np.asarray(mb_l_rewards, dtype=np.float32)
        # mb_terminals = np.asarray(mb_terminals, dtype=np.float32)
        # last_value, last_l_value = policy.predict_values([s_])
        rescale = np.mean([len(path) for path in last_training_paths])



        initial_return = []
        mb_advs = []
        for path in last_training_paths:
            lastgaelam = 0
            path_advs = np.zeros_like(path['rewards'])
            path_values = path['value']
            path_next_values = path['value'][1:]
            path_next_values.append(policy.predict_values(path['obs_'][-1]))
            for t in reversed(range(len(path_values))):

                delta = path['rewards'][t] + gamma * path_next_values[t]*(1- path['done'][t]) - path_values[t]
                path_advs[t] = lastgaelam = delta + gamma * gae_lamda * (1- path['done'][t]) * lastgaelam

            path_returns = path_advs + path_values
            initial_return.append(path_returns[0])
            mb_advs.extend(path_advs)
            mb_obs.extend(path['obs'])
            mb_obs_.extend(path['obs_'])
            mb_values.extend(path['value'])
            mb_terminals.extend(path['done'])
            mb_t.extend(path['t'])
            mb_actions.extend(path['action'])

        initial_return = np.asarray(initial_return, dtype=np.float32)
        mb_obs = np.asarray(mb_obs, dtype=s.dtype)
        mb_values = np.asarray(mb_values, dtype=s.dtype)
        mb_actions = np.asarray(mb_actions, dtype=action.dtype)
        mb_obs_ = np.asarray(mb_obs_, dtype=s_.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_terminals = np.asarray(mb_terminals, dtype=np.float32)
        mb_advs = np.asarray(mb_advs, dtype=np.float32)
        mb_t = np.asarray(mb_t, dtype=np.float32)
        mb_returns = mb_advs + mb_values
        mblossvals = []
        inds = np.arange(len(mb_advs), dtype=int)
        initial_return = np.mean(initial_return)
        # Randomize the indexes
        np.random.shuffle(inds)
        # 0 to batch_size with batch_train_size step
        # if sum(current_path['l_rewards'])>0:
        #     policy.ALPHA3 = min(policy.ALPHA3 * 1.5, policy_params['alpha3'])
        # else:
        #     policy.ALPHA3 = min(policy.ALPHA3 * 1.01, policy_params['alpha3'])


        slices = (arr[inds] for arr in (mb_obs, mb_obs_, mb_returns, mb_advs,mb_actions, mb_values, mb_t))

        # print(**slices)
        mblossvals.append(policy.update(*slices, initial_return, cliprangenow, lr_c_now, rescale))


        mblossvals = np.mean(mblossvals, axis=0)
        frac = 1.0 - (global_step - 1.0) / max_global_steps
        cliprangenow = cliprange*frac
        lr_c_now = lr_c * frac  # learning rate for critic
        # lr_l_now = lr_l * frac  # learning rate for critic


        logger.logkv("total_timesteps", global_step)

        training_diagnotic = evaluate_training_rollouts(last_training_paths)

        if training_diagnotic is not None:
            # [training_diagnotics[key].append(training_diagnotic[key]) for key in training_diagnotic.keys()]\
            eval_diagnotic = training_evaluation(variant, evaluation_env, policy)
            [logger.logkv(key, eval_diagnotic[key]) for key in eval_diagnotic.keys()]
            training_diagnotic.pop('return')
            [logger.logkv(key, training_diagnotic[key]) for key in training_diagnotic.keys()]
            logger.logkv('lr_c', lr_c_now)
            [logger.logkv(name, value) for name, value in zip(policy.diagnosis_names, mblossvals)]
            string_to_print = ['time_step:', str(global_step), '|']
            [string_to_print.extend([key, ':', str(eval_diagnotic[key]), '|'])
             for key in eval_diagnotic.keys()]
            [string_to_print.extend([key, ':', str(round(training_diagnotic[key], 2)), '|'])
             for key in training_diagnotic.keys()]
            print(''.join(string_to_print))
        logger.dumpkvs()
        # 状态更新


        # OUTPUT TRAINING INFORMATION AND LEARNING RATE DECAY





    print('Running time: ', time.time() - t1)
    return
