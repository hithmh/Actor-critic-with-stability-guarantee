import numpy as np
import math
import scipy.linalg as linalg
lqr = linalg.solve_continuous_are
import time
from collections import OrderedDict, deque
import os
from copy import deepcopy
import sys
sys.path.append("..")
import logger
from variant import *


class LQR(object):
    def __init__(self, a_dim, d_dim, s_dim, variant):
        theta_threshold_radians = 20 * 2 * math.pi / 360
        length = 0.5
        masscart = 1
        masspole = 0.1
        total_mass = (masspole + masscart)
        polemass_length = (masspole * length)
        g = 10
        tau = 0.02
        H = np.array([
            [1, 0, 0, 0],
            [0, total_mass, 0, - polemass_length],
            [0, 0, 1, 0],
            [0, - polemass_length, 0, (2 * length) ** 2 * masspole / 3]
        ])

        Hinv = np.linalg.inv(H)

        A = Hinv @ np.array([
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, - polemass_length * g, 0]
        ])
        B = Hinv @ np.array([0, 1.0, 0, 0]).reshape((4, 1))
        Q = np.diag([1/100, 0., 20 *(1/ theta_threshold_radians)**2, 0.])
        R = np.array([[0.1]])

        P = lqr(A, B, Q, R)
        Rinv = np.linalg.inv(R)
        K = Rinv @ B.T @ P
        self.use_Kalman = variant['use_Kalman']
        if self.use_Kalman:
            discrete_A = np.diag(np.ones([4])) + A * tau
            discrete_B = B * tau
            self.filter = Legubger_filter(discrete_A, discrete_B, np.array([[1.,0.,0.,0.],[0.,0.,1.,0.]]))

        self.K = K
        self.u_0 = np.zeros([1])

    def choose_action(self, x, arg):
        if self.use_Kalman:
            x = self.filter.estimate(x, self.u_0)
        x1 = np.copy(x)
        x1[2] = np.sin(x1[2])
        self.u_0 = np.dot(self.K, x)
        return self.u_0

    def reset(self):
        self.u_0 = np.zeros([1])
        if self.use_Kalman:
            self.filter.reset()

    def restore(self, log_path):

        return True

class Legubger_filter(object):

    def __init__(self,A, B, C):
        self.L = np.array([[1.92668388701879, 0.109692525183410],[48.0804019233042, 12.1540057699631],
                           [0.0548643352320299, 2.05876680124335],[5.49508778749990,  62.5683170108460]])

        self.A = A
        self.B = B
        self.C = C
        self.x_0 = np.zeros([4])

    def estimate(self, z, u_0):
        x_hat = self.A.dot(self.x_0) + self.B .dot(u_0) + self.L.dot(z - self.C.dot(self.x_0))
        self.x_0 = x_hat
        z_hat = self.C.dot(x_hat)
        return x_hat

    def reset(self):
        self.x_0 = np.zeros([4])

class Kalman_filter(object):

    def __init__(self,A, B, C):
        self.A = A
        self.B = B
        self.P = np.eye(4)
        self.Q = np.eye(4)
        self.R = np.eye(2)
        self.C = C
        self.x_0 = np.zeros([4])

    def estimate(self, z, u_0):
        x_hat = self.A.dot(self.x_0) + self.B .dot(u_0)
        P = np.dot(self.A .dot(self.P),self.A.T) + self.Q
        K = np.dot(P.dot(self.C.T), np.linalg.inv(np.dot(self.C .dot(P), self.C.T) + self.R))
        x = x_hat + K.dot(z - self.C .dot(x_hat))
        # self.P = (np.eye(4) - K.dot(self.C)).dot(P)
        I = np.eye(4)
        self.P = np.dot(np.dot(I - np.dot(K, self.C), P), (I - np.dot(K, self.C)).T) + np.dot(np.dot(K, self.R), K.T)
        self.x_0 = x
        z_hat = self.C.dot(x)
        return x

    def reset(self):
        self.P = np.eye(4)
        self.x_0 = np.zeros([4])

def eval(variant):
    env_name = variant['env_name']
    env = get_env_from_name(env_name)

    env_params = variant['env_params']


    max_episodes = env_params['max_episodes']
    max_ep_steps = env_params['max_ep_steps']


    alg_name = variant['algorithm_name']
    policy_build_fn = get_policy(alg_name)
    policy_params = variant['alg_params']
    root_path = variant['log_path']

    a_upperbound = env.action_space.high
    a_lowerbound = env.action_space.low
    policy = policy_build_fn(policy_params)
    if 'cartpole' in env_name:
        mag = env_params['impulse_mag']
    # For analyse
    Render = env_params['eval_render']
    # Training setting
    t1 = time.time()
    die_count = 0
    for i in range(variant['num_of_paths']):

        log_path = variant['log_path']+'/eval/' + str(0)
        logger.configure(dir=log_path, format_strs=['csv'])
        s = env.reset()
        if 'Fetch' in env_name or 'Hand' in env_name:
            s = np.concatenate([s[key] for key in s.keys()])

        for j in range(max_ep_steps):
            if Render:
                env.render()
            a = policy.choose_action(s)
            action = a_lowerbound + (a + 1.) * (a_upperbound - a_lowerbound) / 2
            if (j+1)%100 ==0 and 'cartpole'in env_name:

                impulse = mag * np.sign(s[0])
                # print('impulse comming:',impulse)
            # Run in simulator
                s_, r, done, info = env.step(action,impulse=impulse)
            else:
                s_, r, done, info = env.step(action)
            if 'Fetch' in env_name or 'Hand' in env_name:
                s_ = np.concatenate([s_[key] for key in s_.keys()])
                if info['done'] > 0:
                    done = True
            logger.logkv('rewards', r)
            logger.logkv('timestep', j)
            logger.dumpkvs()
            l_r = info['l_rewards']
            if j == max_ep_steps - 1:
                done = True
            s = s_
            if done:
                if j < 199:

                    die_count+=1
                if 'cartpole' in env_name:
                    print('episode:', i,
                          'death:', die_count,
                          'mag:',mag
                          )
                break
    print('Running time: ', time.time() - t1)
    return

if __name__=='__main__':
    lqr_policy = LQR(0.95)
