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
from .utils3 import get_evaluation_rollouts, evaluate_rollouts, evaluate_training_rollouts
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

                 ):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.a_dim, self.s_dim = a_dim, s_dim,


        self.use_lyapunov = args['use_lyapunov']
        self.finite_horizon = args['finite_horizon']
        self.horizon = args['horizon']
        self.form_of_lyapunov = args['form_of_lyapunov']
        self.use_baseline = args['use_baseline']

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
        self.L_UPDATE_STEPS = L_UPDATE_STEPS
        self._verbose_cg = verbose_cg
        self._linesearch_infeasible_recovery = linesearch_infeasible_recovery
        self.expected_safety_threshold = args['safety_threshold']

        self.tfs = tf.placeholder(tf.float32, [None, self.s_dim], 'state')
        # self.cons_S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.tfdc_r = tf.placeholder(tf.float32, [None], 'discounted_r')
        self.target_of_l = tf.placeholder(tf.float32, [None], 'target_of_l')
        self.l_r = tf.placeholder(tf.float32, [None], 'l_r')
        self.safety_threshold = tf.placeholder(tf.float32, [None], 'd_0')
        self.safety_gradient_rescale = tf.placeholder(tf.float32, [None], 'T')
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None], 'old_v')
        self.OLDLPRED = OLDLPRED = tf.placeholder(tf.float32, [None], 'old_l')
        self.alpha3 =  tf.placeholder(tf.float32, [], 'alpha3')
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])
        self.LR_C = tf.placeholder(tf.float32, [], 'LR_C')
        self.v = self._build_c(self.tfs, trainable=True)
        self.l = self._build_l(self.tfs)
        self.LR_L = tf.placeholder(tf.float32, [], 'LR_L')
        self.labda = tf.placeholder(tf.float32, None, 'Labda')
        self.tfa = tf.placeholder(tf.float32, [None, self.a_dim], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None], 'advantage')
        self.constraint_adv = tf.placeholder(tf.float32, [None], 'cons_advantage')
        # l_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Lyapunov')
        self.ALPHA3 = 1e-9
        # ACTOR

        # model_policy = GuassianMLP(self.tfs, self.a_dim, 'model_policy', trainable=True)
        #
        # self.policy = GuassianMLP_from_model(self.tfs, model_policy, 'pi', trainable=True)
        self.policy = GuassianMLP_from_flatten_params(self.tfs, self.a_dim, 'pi', trainable=True)
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
            # self.kl_mean = tf.reduce_mean(tf.reduce_mean(tfp.distributions.kl_divergence(pi,oldpi), axis=1))
            # old_std = tf.exp(self.old_policy.log_sigma)
            # new_std = tf.exp(self.policy.log_sigma)
            # old_means = self.old_policy.mu
            # new_means = self.policy.mu
            # # means: (N*A)
            # # std: (N*A)
            # # formula:
            # # { (\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2 } / (2\sigma_2^2) +
            # # ln(\sigma_2/\sigma_1)
            # numerator = tf.square(old_means - new_means) + tf.square(old_std) - tf.square(new_std)
            # denominator = 2 * tf.square(new_std)
            # self.kl_mean = tf.reduce_mean(tf.reduce_sum(numerator / denominator +
            #                                      self.policy.log_sigma -self.old_policy.log_sigma, axis=-1))
            # numerator = tf.square(old_means - new_means) + tf.square(new_std)
            # denominator = 2 * tf.square(old_std)
            # self.kl_mean = tf.reduce_mean(tf.reduce_sum(numerator / denominator +self.old_policy.log_sigma
            #                                             -self.policy.log_sigma, axis=-1))

        # self.cons_l = self._build_l(self.cons_S,reuse=True)
        self.l_ = self._build_l(self.S_, reuse=True)

        alpha3 = self.alpha3

        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                # ratio =tf.reduce_mean(pi.prob(self.tfa) / oldpi.prob(self.tfa), axis=1)
                self.loss = -tf.reduce_mean(ratio * self.tfadv)
                self.ratio = ratio
            if self.use_lyapunov:
                # self.safety_threshold = 0
                self.constraint_func = tf.reduce_mean(ratio * self.l_ - self.l + alpha3 * self.l_r)
                self.pre_compute_safety_eval = False
            else:
                # self.safety_threshold = tf.reduce_max(args['safety_threshold']-self.target_of_l)
                self.constraint_func = self.safety_gradient_rescale*tf.reduce_mean(ratio * self.constraint_adv)
                self.pre_compute_safety_eval = True

        f_grads = tf.gradients(self.loss, pi_params)
        self.f_grads = f_grads
        cons_grads = tf.gradients(self.constraint_func, pi_params)
        self.cons_grads = cons_grads
        self.Heassian_obj = Heassian_for_flat_params(self.kl_mean, self.policy, self.reg_coeff, self.sess)



        # CRITIC
        self.advantage = self.tfdc_r - self.v
        # self.closs = tf.reduce_mean(tf.square(self.advantage))
        vpredclipped = OLDVPRED + tf.clip_by_value(self.v - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        # Unclipped value
        vf_losses1 = tf.square(self.tfdc_r - self.v)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - self.tfdc_r)
        self.closs = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        self.ctrain = tf.train.AdamOptimizer(self.LR_C).minimize(self.closs)

        #Lyapunov
        # self.lloss = tf.losses.mean_squared_error(labels=tf.expand_dims(self.target_of_l, axis=1), predictions=self.l)
        OLDLPRED = tf.expand_dims(OLDLPRED, axis=1)
        target_of_l = tf.expand_dims(self.target_of_l, axis=1)
        lpredclipped = OLDLPRED + tf.clip_by_value(self.l - OLDLPRED, - CLIPRANGE, CLIPRANGE)
        # Unclipped value
        lf_losses1 = tf.losses.mean_squared_error(labels=target_of_l, predictions=self.l)
        # Clipped value
        lf_losses2 = tf.losses.mean_squared_error(labels=target_of_l, predictions=lpredclipped)
        self.lloss = tf.reduce_mean(tf.maximum(lf_losses1, lf_losses2))
        self.ltrain = tf.train.AdamOptimizer(self.LR_L).minimize(self.lloss)

        self.diagnosis_names = ['value_loss', 'lyapunov_loss', 'policy_entropy']
        self.diagnosis = [self.closs, self.lloss, self.entropy]
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
        s = s[np.newaxis, :]
        a, v, l_v = self.sess.run([self.sample_op, self.v, self.l], {self.tfs: s})
        if evaluation:
            action = np.tanh(a[0])
            return action
        else:
            return a, v[0], l_v[0]
    # def choose_action(self, s, evaluation=False):
    #     if evaluation:
    #         s = np.squeeze(s)
    #     s = s[np.newaxis, :]
    #     a, v = self.sess.run([self.sample_op, self.v], {self.tfs: s})
    #     if evaluation:
    #         action = np.tanh(a[0])
    #         return action
    #     else:
    #         return a, v[0]
    def predict_values(self, s):

        return self.sess.run([self.v, self.l], {self.tfs: s})

    def update(self, s, s_, returns, l_returns, advs, l_advs, a, old_values, old_l_values, l_reward, finite_safety_values,
               safety_eval,cliprangenow, LR_C, LR_L, length_of_trajectory):
        # advs = advs[:, np.newaxis]
        # l_advs = l_advs[:, np.newaxis]
        # returns = returns[:, np.newaxis]
        # l_returns = l_returns[:, np.newaxis]
        # l_reward = l_reward[:, np.newaxis]
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        l_advs = (l_advs - l_advs.mean()) / (l_advs.std() + 1e-8)
        inputs = [s, s_, returns, l_returns, advs, l_advs, a, old_values, old_l_values, finite_safety_values, l_reward,]

        def subsampled_inputs(inputs,subsample_grouped_inputs = None):
            if self._subsample_factor < 1:
                if subsample_grouped_inputs is None:
                    subsample_grouped_inputs = [inputs]
                subsample_inputs = tuple()
                for inputs_grouped in subsample_grouped_inputs:
                    n_samples = len(inputs_grouped[0])
                    inds = np.random.choice(
                        n_samples, int(n_samples * self._subsample_factor), replace=False)
                    subsample_inputs += tuple([x[inds] for x in inputs_grouped])
            else:
                subsample_inputs = inputs
            return subsample_inputs

        subsampled_data = subsampled_inputs(inputs)
        sub_s = subsampled_data[0]
        sub_s_ = subsampled_data[1]
        sub_returns = subsampled_data[2]
        sub_l_returns = subsampled_data[3]
        sub_advs = subsampled_data[4]
        sub_l_advs = subsampled_data[5]
        sub_a = subsampled_data[6]

        sub_old_l_value = subsampled_data[8]
        sub_finite_safety_values = subsampled_data[-2]
        sub_l_reward = subsampled_data[-1]

        if self.use_lyapunov:
            threshold = [0]
        else:
            threshold = [np.max(self.expected_safety_threshold-l_returns,0)]

        feed_dict = {self.tfs: s, self.tfa: a, self.tfadv: advs, self.tfdc_r:returns, self.LR_C: LR_C, self.LR_L: LR_L,
                     self.S_:s_, self.OLDVPRED:old_values, self.CLIPRANGE:cliprangenow, self.safety_gradient_rescale:[length_of_trajectory,],
                     self.safety_threshold:threshold, self.alpha3:self.ALPHA3}
        sub_feed_dict = {self.tfs: sub_s, self.tfa: sub_a, self.tfadv: sub_advs, self.tfdc_r: sub_returns, self.LR_C: LR_C, self.LR_L: LR_L,
                     self.S_: sub_s_,self.safety_gradient_rescale:[length_of_trajectory],self.alpha3:self.ALPHA3,
                         self.safety_threshold: threshold}
        if self.use_lyapunov:
            if self.form_of_lyapunov == 'l_value':
                if self.finite_horizon:
                    feed_dict.update({self.target_of_l: l_returns, self.l_r: l_reward})
                    sub_feed_dict.update({self.target_of_l: sub_l_returns, self.l_r: sub_l_reward})
                else:
                    feed_dict.update({self.target_of_l: finite_safety_values, self.l_r: l_reward})
                    sub_feed_dict.update({self.target_of_l: sub_finite_safety_values, self.l_r: sub_l_reward})
            elif self.form_of_lyapunov == 'l_reward':
                feed_dict.update({self.target_of_l: l_reward, self.l_r: l_reward})
                sub_feed_dict.update({self.target_of_l: sub_l_reward, self.l_r: sub_l_reward})
        else:
            feed_dict.update({self.target_of_l: l_returns, self.constraint_adv: l_advs, self.OLDLPRED: old_l_values})
            sub_feed_dict.update({self.target_of_l: sub_l_returns, self.constraint_adv: sub_l_advs, self.OLDLPRED: sub_old_l_value})

        self.sess.run(self.update_oldpi_op)
        # update Lyapunov
        [self.sess.run(self.ltrain, feed_dict) for _ in range(self.L_UPDATE_STEPS)]
        if self.use_baseline:
            self.optimize_baseline(feed_dict, sub_feed_dict)
        else:
            self.optimize(feed_dict, sub_feed_dict, safety_eval)

        # update critic
        [self.sess.run(self.ctrain, feed_dict) for _ in range(self.C_UPDATE_STEPS)]

        return self.sess.run(self.diagnosis, feed_dict)



    def optimize(self,feed_dict, sub_feed_dict, safety_eval):

        loss_before = self.sess.run(self.loss,feed_dict)

        flat_g = self.sess.run(self.f_grads, feed_dict)[0]
        flat_b = self.sess.run(self.cons_grads, feed_dict)[0]
        # Hx = self.Heassian_obj.build_eval(sub_feed_dict)
        Hx = self.Heassian_obj.build_eval(feed_dict)

        norm_g = np.sqrt(flat_g.dot(flat_g))
        unit_g = flat_g / norm_g
        v = norm_g * cg(Hx, unit_g, cg_iters=self._cg_iters, verbose=self._verbose_cg)
        approx_g = Hx(v)
        # approx_g = f_Ax(H, v, params_shape)
        q = v.dot(approx_g)  # approx = g^T H^{-1} g
        delta = 2 * self._max_quad_constraint_val

        eps = 1e-8
        residual = np.sqrt((approx_g - flat_g).dot(approx_g - flat_g))
        rescale = q / (v.dot(v))
        logger.record_tabular("OptimDiagnostic_Residual", residual)
        logger.record_tabular("OptimDiagnostic_Rescale", rescale)
        if self.pre_compute_safety_eval:
            S = safety_eval
        else:
            S = self.sess.run(self.constraint_func, feed_dict)
        c = S - self.sess.run(self.safety_threshold, feed_dict)
        if c > 0:
            logger.log("warning! safety constraint is already violated")
        else:
            # the current parameters constitute a feasible point: save it as "last good point"
            self.last_safe_point = np.copy(self.policy.get_params_values(self.sess))

        # can't stop won't stop (unless something in the conditional checks / calculations that follow
        # require premature stopping of optimization process)
        stop_flag = False

        if flat_b.dot(flat_b) <= eps:
            # if safety gradient is zero, linear constraint is not present;
            # ignore its implementation.
            lam = np.sqrt(q / delta)
            nu = 0
            w = 0
            r, s, A, B = 0, 0, 0, 0
            optim_case = 4
        else:
            # if self._resample_inputs:
            #     Hx = self._hvp_approach.build_eval(subsample_inputs2 + extra_inputs)

            norm_b = np.sqrt(flat_b.dot(flat_b))
            unit_b = flat_b / norm_b
            w = norm_b * cg(Hx, unit_b, cg_iters=self._cg_iters, verbose=self._verbose_cg)
            # w = norm_b * cg_try(H, unit_b, params_shape, cg_iters=self._cg_iters, verbose=self._verbose_cg)
            r = w.dot(approx_g)  # approx = b^T H^{-1} g
            s = w.dot(Hx(w))  # approx = b^T H^{-1} b
            # s = w.dot(f_Ax(H, v, params_shape))
            # figure out lambda coeff (lagrange multiplier for trust region)
            # and nu coeff (lagrange multiplier for linear constraint)
            A = q - r ** 2 / s  # this should always be positive by Cauchy-Schwarz
            B = delta - c ** 2 / s  # this one says whether or not the closest point on the plane is feasible

            # if (B < 0), that means the trust region plane doesn't intersect the safety boundary

            if c < 0 and B < 0:
                # point in trust region is feasible and safety boundary doesn't intersect
                # ==> entire trust region is feasible
                optim_case = 3
            elif c < 0 and B > 0:
                # x = 0 is feasible and safety boundary intersects
                # ==> most of trust region is feasible
                optim_case = 2
            elif c > 0 and B > 0:
                # x = 0 is infeasible (bad! unsafe!) and safety boundary intersects
                # ==> part of trust region is feasible
                # ==> this is 'recovery mode'
                optim_case = 1
                if self.attempt_feasible_recovery:
                    logger.log("alert! conjugate constraint optimizer is attempting feasible recovery")
                else:
                    logger.log(
                        "alert! problem is feasible but needs recovery, and we were instructed not to attempt recovery")
                    stop_flag = True
            else:
                # x = 0 infeasible (bad! unsafe!) and safety boundary doesn't intersect
                # ==> whole trust region infeasible
                # ==> optimization problem infeasible!!!
                optim_case = 0
                if self.attempt_infeasible_recovery:
                    logger.log("alert! conjugate constraint optimizer is attempting infeasible recovery")
                else:
                    logger.log("alert! problem is infeasible, and we were instructed not to attempt recovery")
                    stop_flag = True

            # default dual vars, which assume safety constraint inactive
            # (this corresponds to either optim_case == 3,
            #  or optim_case == 2 under certain conditions)
            lam = np.sqrt(q / delta)
            nu = 0

            if optim_case == 2 or optim_case == 1:

                # dual function is piecewise continuous
                # on region (a):
                #
                #   L(lam) = -1/2 (A / lam + B * lam) - r * c / s
                #
                # on region (b):
                #
                #   L(lam) = -1/2 (q / lam + delta * lam)
                #

                lam_mid = r / c

                L_mid = - 0.5 * (q / lam_mid + lam_mid * delta)

                if A / (B + eps) <0.:
                    print('A is negative')
                lam_a = np.sqrt(A / (B + eps))
                if A * B <0.:
                    print('A is negative')
                L_a = -np.sqrt(A * B) - r * c / (s + eps)


                # note that for optim_case == 1 or 2, B > 0, so this calculation should never be an issue

                lam_b = np.sqrt(q / delta)
                L_b = -np.sqrt(q * delta)

                # those lam's are solns to the pieces of piecewise continuous dual function.
                # the domains of the pieces depend on whether or not c < 0 (x=0 feasible),
                # and so projection back on to those domains is determined appropriately.
                if lam_mid > 0:
                    if c < 0:
                        # here, domain of (a) is [0, lam_mid)
                        # and domain of (b) is (lam_mid, infty)
                        if lam_a > lam_mid:
                            lam_a = lam_mid
                            L_a = L_mid
                        if lam_b < lam_mid:
                            lam_b = lam_mid
                            L_b = L_mid
                    else:
                        # here, domain of (a) is (lam_mid, infty)
                        # and domain of (b) is [0, lam_mid)
                        if lam_a < lam_mid:
                            lam_a = lam_mid
                            L_a = L_mid
                        if lam_b > lam_mid:
                            lam_b = lam_mid
                            L_b = L_mid

                    if L_a >= L_b:
                        lam = lam_a
                    else:
                        lam = lam_b

                else:
                    if c < 0:
                        lam = lam_b
                    else:
                        lam = lam_a

                nu = max(0, lam * c - r) / (s + eps)

        logger.record_tabular("OptimCase", optim_case)  # 4 / 3: trust region totally in safe region;
        # 2 : trust region partly intersects safe region, and current point is feasible
        # 1 : trust region partly intersects safe region, and current point is infeasible
        # 0 : trust region does not intersect safe region
        logger.record_tabular("LagrangeLamda", lam)  # dual variable for trust region
        logger.record_tabular("LagrangeNu", nu)  # dual variable for safety constraint
        logger.record_tabular("OptimDiagnostic_q", q)  # approx = g^T H^{-1} g
        logger.record_tabular("OptimDiagnostic_r", r)  # approx = b^T H^{-1} g
        logger.record_tabular("OptimDiagnostic_s", s)  # approx = b^T H^{-1} b
        logger.record_tabular("OptimDiagnostic_c", c)  # if > 0, constraint is violated
        logger.record_tabular("OptimDiagnostic_A", A)
        logger.record_tabular("OptimDiagnostic_B", B)
        logger.record_tabular("OptimDiagnostic_S", S)
        if nu == 0:
            logger.log("safety constraint is not active!")

        # Predict worst-case next S
        nextS = S + np.sqrt(delta * s)
        logger.record_tabular("OptimDiagnostic_WorstNextS", nextS)

        # for cases where we will not attempt recovery, we stop here. we didn't stop earlier
        # because first we wanted to record the various critical quantities for understanding the failure mode
        # (such as optim_case, B, c, S). Also, the logger gets angry if you are inconsistent about recording
        # a given quantity from iteration to iteration. That's why we have to record a BacktrackIters here.
        def record_zeros():
            logger.record_tabular("BacktrackIters", 0)
            logger.record_tabular("LossRejects", 0)
            logger.record_tabular("QuadRejects", 0)
            logger.record_tabular("LinRejects", 0)

        if optim_case > 0:
            flat_descent_step = (1. / (lam + eps)) * (v + nu * w)
        else:
            # current default behavior for attempting infeasible recovery:
            # take a step on natural safety gradient
            flat_descent_step = np.sqrt(delta / (s + eps)) * w

        logger.log("descent direction computed")

        prev_param = np.copy(self.policy.get_params_values(self.sess))

        prev_lin_constraint_val = S
        logger.record_tabular("PrevLinConstVal", prev_lin_constraint_val)

        lin_reject_threshold = self.sess.run(self.safety_threshold, feed_dict)

        logger.record_tabular("LinRejectThreshold", lin_reject_threshold)

        def check_nan():
            loss, quad_constraint_val, lin_constraint_val = \
                self.sess.run([self.loss, self.kl_mean, self.constraint_func], feed_dict)
            if np.isnan(loss) or np.isnan(quad_constraint_val) or np.isnan(lin_constraint_val):
                logger.log("Something is NaN. Rejecting the step!")
                if np.isnan(loss):
                    logger.log("Violated because loss is NaN")
                if np.isnan(quad_constraint_val):
                    logger.log("Violated because quad_constraint %s is NaN" %
                               self._constraint_name_1)
                if np.isnan(lin_constraint_val):
                    logger.log("Violated because lin_constraint %s is NaN" %
                               self._constraint_name_2)
                self.policy.set_params(prev_param, self.sess)

        def line_search(check_loss=True, check_quad=True, check_lin=True):
            loss_rejects = 0
            quad_rejects = 0
            lin_rejects = 0
            n_iter = 0
            # norm_decent = np.sqrt(flat_descent_step.dot(flat_descent_step))
            # unit_decent = flat_descent_step/norm_decent
            for n_iter, ratio in enumerate(self._backtrack_ratio ** np.arange(self._max_backtracks)):
                cur_step = ratio * flat_descent_step
                cur_param = prev_param - cur_step
                self.policy.set_params(cur_param, self.sess)
                loss, quad_constraint_val, lin_constraint_val = \
                    self.sess.run([self.loss, self.kl_mean,self.constraint_func], feed_dict)
                loss_flag = loss < loss_before
                quad_flag = quad_constraint_val <= self._max_quad_constraint_val
                lin_flag = lin_constraint_val <= lin_reject_threshold
                if check_loss and not (loss_flag):
                    logger.log("At backtrack itr %i, loss failed to improve." % n_iter)
                    loss_rejects += 1
                if check_quad and not (quad_flag):
                    logger.log("At backtrack itr %i, quad constraint violated." % n_iter)
                    logger.log("Quad constraint violation was %.3f %%." % (
                                100 * (quad_constraint_val / self._max_quad_constraint_val) - 100))
                    quad_rejects += 1
                if check_lin and not (lin_flag):
                    logger.log("At backtrack itr %i, expression for lin constraint failed to improve." % n_iter)
                    logger.log("Lin constraint violation was %.3f ." % (
                                lin_constraint_val - lin_reject_threshold))
                    lin_rejects += 1

                if (loss_flag or not (check_loss)) and (quad_flag or not (check_quad)) and (
                        lin_flag or not (check_lin)):
                    logger.log("Accepted step at backtrack itr %i." % n_iter)
                    break

            logger.record_tabular("BacktrackIters", n_iter)
            logger.record_tabular("LossRejects", loss_rejects)
            logger.record_tabular("QuadRejects", quad_rejects)
            logger.record_tabular("LinRejects", lin_rejects)
            return loss, quad_constraint_val, lin_constraint_val, n_iter

        def wrap_up():
            if optim_case < 4:
                lin_constraint_val = self.sess.run([self.constraint_func], feed_dict)
                lin_constraint_delta = lin_constraint_val - prev_lin_constraint_val
                logger.record_tabular("LinConstraintDelta", lin_constraint_delta)

                cur_param = self.policy.get_params_values(self.sess)

                next_linear_S = S + flat_b.dot(cur_param - prev_param)
                next_surrogate_S = S + lin_constraint_delta

                lin_surrogate_acc = 100. * (next_linear_S - next_surrogate_S) / next_surrogate_S

                logger.record_tabular("PredictedLinearS", next_linear_S)
                logger.record_tabular("PredictedSurrogateS", next_surrogate_S)
                logger.record_tabular("LinearSurrogateErr", lin_surrogate_acc)

                lin_pred_err = (self._last_lin_pred_S - S)  # / (S + eps)
                surr_pred_err = (self._last_surr_pred_S - S)  # / (S + eps)
                logger.record_tabular("PredictionErrorLinearS", lin_pred_err)
                logger.record_tabular("PredictionErrorSurrogateS", surr_pred_err)
                self._last_lin_pred_S = next_linear_S
                self._last_surr_pred_S = next_surrogate_S

            else:
                logger.record_tabular("LinConstraintDelta", 0)
                logger.record_tabular("PredictedLinearS", 0)
                logger.record_tabular("PredictedSurrogateS", 0)
                logger.record_tabular("LinearSurrogateErr", 0)

                lin_pred_err = (self._last_lin_pred_S - 0)  # / (S + eps)
                surr_pred_err = (self._last_surr_pred_S - 0)  # / (S + eps)
                logger.record_tabular("PredictionErrorLinearS", lin_pred_err)
                logger.record_tabular("PredictionErrorSurrogateS", surr_pred_err)
                self._last_lin_pred_S = 0
                self._last_surr_pred_S = 0

        if stop_flag == True:
            record_zeros()
            wrap_up()
            return

        if optim_case == 1 and not (self.revert_to_last_safe_point):
            if self._linesearch_infeasible_recovery:
                logger.log(
                    "feasible recovery mode: constrained natural gradient step. performing linesearch on constraints.")
                line_search(False, True, True)
            else:
                self.policy.set_params(prev_param - flat_descent_step, self.sess)
                logger.log("feasible recovery mode: constrained natural gradient step. no linesearch performed.")
                record_zeros()
            check_nan()

            wrap_up()
            return
        elif optim_case == 0 and not (self.revert_to_last_safe_point):
            if self._linesearch_infeasible_recovery:
                logger.log("infeasible recovery mode: natural safety step. performing linesearch on constraints.")
                # line_search(False, True, True)
                line_search(False, True, True)
            else:
                self.policy.set_params(prev_param - flat_descent_step, self.sess)
                logger.log("infeasible recovery mode: natural safety gradient step. no linesearch performed.")
                record_zeros()
            check_nan()

            wrap_up()
            return
        elif (optim_case == 0 or optim_case == 1) and self.revert_to_last_safe_point:
            if self.last_safe_point:
                self.policy.set_params(self.last_safe_point, self.sess)
                logger.log("infeasible recovery mode: reverted to last safe point!")
            else:
                logger.log("alert! infeasible recovery mode failed: no last safe point to revert to.")
                record_zeros()
            wrap_up()
            return

        loss, quad_constraint_val, lin_constraint_val, n_iter = line_search()

        if (np.isnan(loss) or np.isnan(quad_constraint_val) or np.isnan(lin_constraint_val) or loss >= loss_before
            or quad_constraint_val >= self._max_quad_constraint_val
            or lin_constraint_val > lin_reject_threshold) and not self._accept_violation:
            logger.log("Line search condition violated. Rejecting the step!")
            if np.isnan(loss):
                logger.log("Violated because loss is NaN")
            if np.isnan(quad_constraint_val):
                logger.log("Violated because quad_constraint %s is NaN" %
                           self._constraint_name_1)
            if np.isnan(lin_constraint_val):
                logger.log("Violated because lin_constraint %s is NaN" %
                           self._constraint_name_2)
            if loss >= loss_before:
                logger.log("Violated because loss not improving")
            if quad_constraint_val >= self._max_quad_constraint_val:
                logger.log(
                    "Violated because constraint %s is violated" % self._constraint_name_1)
            if lin_constraint_val > lin_reject_threshold:
                logger.log(
                    "Violated because constraint %s exceeded threshold" % self._constraint_name_2)
            self.policy.set_params(prev_param, self.sess)
        logger.log("backtrack iters: %d" % n_iter)
        logger.log("computing loss after")
        logger.log("optimization finished")
        wrap_up()

    def optimize_baseline(self, feed_dict, sub_feed_dict):

        logger.log("computing loss before")
        loss_before = self.sess.run(self.loss, feed_dict)
        logger.log("performing update")
        logger.log("computing descent direction")

        flat_g = self.sess.run(self.f_grads, feed_dict)[0]
        norm_g = np.sqrt(flat_g.dot(flat_g))
        unit_g = flat_g/norm_g
        Hx = self.Heassian_obj.build_eval(sub_feed_dict)
        # descent_direction = cg(Hx, flat_g, cg_iters=self._cg_iters, verbose=True)
        descent_direction = norm_g*cg(Hx, unit_g, cg_iters=self._cg_iters, verbose=self._verbose_cg)
        # descent_direction = norm_g * cg(Hx, unit_g, cg_iters=50, verbose=self._verbose_cg)
        approx_g = Hx(descent_direction)
        q = descent_direction.dot(approx_g)
        residual = np.sqrt((approx_g - flat_g).dot(approx_g - flat_g))
        rescale = q / (descent_direction.dot(descent_direction))
        logger.record_tabular("OptimDiagnostic_Residual", residual)
        logger.record_tabular("OptimDiagnostic_Rescale", rescale)

        # initial_step_size = np.sqrt(
        #     2.0 * self._max_quad_constraint_val *
        #     (1. / (descent_direction.dot(Hx(descent_direction)) + 1e-8))
        # )

        initial_step_size = np.sqrt(
            2.0 * self._max_quad_constraint_val *
            (1. / (descent_direction.dot(flat_g) + 1e-8))
        )
        if np.isnan(initial_step_size):
            initial_step_size = 1.
        flat_descent_step = initial_step_size * descent_direction
        # norm_descent = np.linalg.norm(flat_descent_step)
        # unit_descent = flat_descent_step/norm_descent
        logger.log("descent direction computed")

        prev_param = np.copy(self.policy.get_params_values(self.sess))
        n_iter = 0
        for n_iter, ratio in enumerate(self._backtrack_ratio ** np.arange(self._max_backtracks)):
            cur_step = ratio * flat_descent_step
            cur_param = prev_param - cur_step
            self.policy.set_params(cur_param, self.sess)
            loss, constraint_val = self.sess.run([self.loss, self.kl_mean], feed_dict)
            if loss < loss_before and constraint_val <= self._max_quad_constraint_val:
                break
        if (np.isnan(loss) or np.isnan(constraint_val) or loss >= loss_before or constraint_val >=
            self._max_quad_constraint_val) and not self._accept_violation:
            logger.log("Line search condition violated. Rejecting the step!")
            if np.isnan(loss):
                logger.log("Violated because loss is NaN")
            if np.isnan(constraint_val):
                logger.log("Violated because constraint %s is NaN" %
                           self._constraint_name_1)
            if loss >= loss_before:
                logger.log("Violated because loss not improving")
            if constraint_val >= self._max_quad_constraint_val:
                logger.log(
                    "Violated because constraint %s is violated" % self._constraint_name_1)
            self.policy.set_params(prev_param, self.sess)
        logger.log("backtrack iters: %d" % n_iter)
        logger.log("computing loss after")
        logger.log("optimization finished")

    #critic模块
    def _build_c(self, s,trainable):
        with tf.variable_scope('Critic'):
            net_0 = tf.layers.dense(s, 256, activation=tf.nn.relu, name='l0', trainable=trainable)
            net_1 = tf.layers.dense(net_0, 128, activation=tf.nn.relu, name='l1', trainable=trainable)
            # net_2 = tf.layers.dense(net_1, 256, activation=tf.nn.relu, name='l2', trainable=trainable)
            # net_3 = tf.layers.dense(net_2, 128, activation=tf.nn.relu, name='l3', trainable=trainable)
            return tf.layers.dense(net_1, 1, trainable=trainable)  # V(s)
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

    evaluation_env = get_env_from_name(env_name)
    env_params = variant['env_params']


    max_ep_steps = env_params['max_ep_steps']
    max_global_steps = env_params['max_global_steps']
    store_last_n_paths = variant['num_of_training_paths']


    alg_name = variant['algorithm_name']
    policy_build_fn = get_policy(alg_name)
    policy_params = variant['alg_params']
    batch_size = policy_params['batch_size']
    sample_steps = batch_size


    lr_c, lr_l = policy_params['lr_c'], policy_params['lr_l']
    cliprange = policy_params['cliprange']
    cliprangenow = cliprange
    lr_c_now = lr_c  # learning rate for critic
    lr_l_now = lr_l  # learning rate for critic
    gamma = policy_params['gamma']
    gae_lamda = policy_params['gae_lamda']
    safety_gae_lamda = policy_params['safety_gae_lamda']

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
    last_training_paths = deque(maxlen=store_last_n_paths)

    s = env.reset()
    if 'Fetch' in env_name or 'Hand' in env_name:
        s = np.concatenate([s[key] for key in s.keys()])
    local_step = 0
    current_path = {'rewards': [],
                    'l_rewards': [],
                    'obs': [],
                    'obs_': [],
                    't':[],
                    'done': [],

                    }
    for j in range(int(np.ceil(max_global_steps / sample_steps))):

        if global_step > max_global_steps:
            break

        mb_obs, mb_obs_, mb_rewards, mb_l_rewards, mb_actions, mb_values, mb_l_values, mb_terminals = [], [], [], [], [], [], [], []

        # For n in range number of steps
        for _ in range(sample_steps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init

            [a], [value], [l_value] = policy.choose_action(s)
            action = np.tanh(a)
            action = a_lowerbound + (action + 1.) * (a_upperbound - a_lowerbound) / 2

            # Run in simulator
            s_, r, done, info = env.step(action)
            if 'Fetch' in env_name or 'Hand' in env_name:
                s_ = np.concatenate([s_[key] for key in s_.keys()])
            local_step +=1
            if local_step == max_ep_steps - 1:
                done = True
            terminal = 1. if done else 0.


            mb_obs.append(s)
            mb_values.append(value)
            mb_l_values.append(l_value)
            mb_actions.append(a)
            mb_terminals.append(terminal)
            mb_obs_.append(s_)
            mb_rewards.append(0)
            mb_l_rewards.append(r)


            if Render:
                env.render()

            current_path['rewards'].append(r)
            current_path['l_rewards'].append(r)
            current_path['obs'].append(s)
            current_path['t'].append(local_step)
            current_path['obs_'].append(s_)
            current_path['done'].append(terminal)
            if done:

                local_step = 0
                last_training_paths.appendleft(current_path)
                current_path = {'rewards': [],
                                'l_rewards': [],
                                'obs': [],
                                'obs_': [],
                                't':[],
                                'done': [],
                                }
                s = env.reset()
                if 'Fetch' in env_name or 'Hand' in env_name:
                    s = np.concatenate([s[key] for key in s.keys()])

            else:
                s = s_
        mb_obs = np.asarray(mb_obs, dtype=s.dtype)
        mb_values = np.asarray(mb_values, dtype=s.dtype)
        mb_l_values = np.asarray(mb_l_values, dtype=s.dtype)
        mb_actions = np.asarray(mb_actions, dtype=action.dtype)
        mb_obs_ = np.asarray(mb_obs_, dtype=s_.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_l_rewards = np.asarray(mb_l_rewards, dtype=np.float32)
        mb_terminals = np.asarray(mb_terminals, dtype=np.float32)
        last_value, last_l_value = policy.predict_values([s_])
        rescale = np.mean([len(path) for path in last_training_paths])
        global_step += sample_steps


        safety_evals = []
        for path in last_training_paths:
            lastgaelam = 0
            path_l_advs = np.zeros_like(path['l_rewards'])
            _, path_l_values = policy.predict_values(path['obs'])
            _, path_next_l_values = policy.predict_values(path['obs_'])
            for t in reversed(range(len(path_l_advs))):
                try:
                    delta = path['l_rewards'][t] + gamma * path_next_l_values[t]*(1- path['done'][t]) - path_l_values[t]
                except IndexError:
                    return
                path_l_advs[t] = lastgaelam = delta + gamma * safety_gae_lamda * (1- path['done'][t]) * lastgaelam

            path_l_returns = path_l_advs + np.squeeze(path_l_values)
            safety_evals.append(path_l_returns[0])
        safety_eval = np.mean(safety_evals)



        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        nextnonterminal = np.ones_like(mb_terminals) - mb_terminals
        # delta = mb_rewards + gamma * mb_values[1:] * nextnonterminal - mb_values[:-1]
        for t in reversed(range(sample_steps)):
            if t == sample_steps-1:
                next_value = last_value
            else:
                next_value = mb_values[t + 1]
            delta = mb_rewards[t] + gamma * next_value * nextnonterminal[t] - mb_values[t]
            # mb_advs[t] = delta
            mb_advs[t] = lastgaelam = delta + gamma * gae_lamda * nextnonterminal[t] * lastgaelam
        mb_returns = mb_advs + mb_values

        mb_l_advs = np.zeros_like(mb_l_rewards)
        lastgaelam = 0
        for t in reversed(range(sample_steps)):
            if t == sample_steps-1:
                next_l_value = last_l_value
            else:
                next_l_value = mb_l_values[t + 1]
            delta = mb_l_rewards[t] + gamma * next_l_value * nextnonterminal[t] - mb_l_values[t]
            mb_l_advs[t] = lastgaelam = delta + gamma * safety_gae_lamda * nextnonterminal[t] * lastgaelam
        mb_l_returns = mb_l_advs + mb_l_values

        if policy.finite_horizon and policy.form_of_lyapunov=='l_value':
            r = deepcopy(mb_l_rewards)
            terminal_index=[-1]
            [terminal_index.append(i) for i,x in enumerate(mb_terminals) if x ==1.]
            if mb_terminals[-1] != 1.:
                terminal_index.append(len(r)-1)

            mb_finite_safety_value = []

            for t_0, t_1 in zip(terminal_index[:-1],terminal_index[1:]):

                path = r[t_0+1:t_1+1]
                path_length = len(path)
                try:
                    last_r = path[-1]
                except IndexError:
                    continue
                path = np.concatenate((path, last_r * np.ones([policy.horizon])), axis=0)
                safety_value = []
                [safety_value.append(path[i:i + policy.horizon].sum()) for i in range(path_length)]
                mb_finite_safety_value.extend(safety_value)

            mb_finite_safety_value = np.array(mb_finite_safety_value)
        else:
            mb_finite_safety_value = np.zeros_like(mb_values)


        mblossvals = []
        inds = np.arange(sample_steps, dtype=int)

        # Randomize the indexes
        np.random.shuffle(inds)
        # 0 to batch_size with batch_train_size step
        if sum(current_path['l_rewards'])>0:
            policy.ALPHA3 = min(policy.ALPHA3 * 1.5, policy_params['alpha3'])
        else:
            policy.ALPHA3 = min(policy.ALPHA3 * 1.01, policy_params['alpha3'])


        slices = (arr[inds] for arr in (mb_obs, mb_obs_, mb_returns, mb_l_returns, mb_advs,
                                          mb_l_advs, mb_actions, mb_values, mb_l_values,
                                          mb_l_rewards, mb_finite_safety_value))

        # print(**slices)
        mblossvals.append(policy.update(*slices, safety_eval, cliprangenow, lr_c_now, lr_l_now, rescale))


        mblossvals = np.mean(mblossvals, axis=0)
        frac = 1.0 - (global_step - 1.0) / max_global_steps
        cliprangenow = cliprange*frac
        lr_c_now = lr_c * frac  # learning rate for critic
        lr_l_now = lr_l * frac  # learning rate for critic

        logger.logkv("total_timesteps", global_step)
        training_diagnotic = evaluate_training_rollouts(last_training_paths)
        if training_diagnotic is not None:
            # [training_diagnotics[key].append(training_diagnotic[key]) for key in training_diagnotic.keys()]\
            # eval_diagnotic = training_evaluation(variant, evaluation_env, policy)
            # [logger.logkv(key, eval_diagnotic[key]) for key in eval_diagnotic.keys()]
            # training_diagnotic.pop('return')
            [logger.logkv(key, training_diagnotic[key]) for key in training_diagnotic.keys()]
            logger.logkv('lr_c', lr_c_now)
            [logger.logkv(name, value) for name, value in zip(policy.diagnosis_names, mblossvals)]
            [training_diagnotic.update({name:value}) for name, value in zip(policy.diagnosis_names, mblossvals)]
            string_to_print = ['time_step:', str(global_step), '|']
            # [string_to_print.extend([key, ':', str(eval_diagnotic[key]), '|'])
            #  for key in eval_diagnotic.keys()]
            [string_to_print.extend([key, ':', str(round(training_diagnotic[key], 2)), '|'])
             for key in training_diagnotic.keys()]
            print(''.join(string_to_print))
        logger.dumpkvs()
        # 状态更新


        # OUTPUT TRAINING INFORMATION AND LEARNING RATE DECAY

    policy.save_result(log_path)



    print('Running time: ', time.time() - t1)
    return
