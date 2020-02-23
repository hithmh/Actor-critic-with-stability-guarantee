"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import csv
import matplotlib.pyplot as plt

class oscillator(gym.Env):

    def __init__(self):
        self.K = 1  # + np.random.uniform(0, 10, 1)
        self.c1 = 1.6
        self.c2 = 0.16
        self.c3 = 0.16
        self.c4 = 0.06
        # self.c1 = 1.6 + np.random.uniform(-1.5,10,1)
        # self.c2 = 0.16 + np.random.uniform(-0.08,0.08,1)
        # self.c3 = 0.16 + np.random.uniform(-0.08,0.08,1)
        # self.c4 = 0.06 + np.random.uniform(-0.03,0.03,1)
        self.b1 = 5
        self.b2 = 5
        self.b3 = 5
        self.b4 = 5
        self.dt = 1.
        self.t = 0
        self.sigma = 0
        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        self.action_space = spaces.Box(low=np.array([0., 0., 0., 0.]), high=np.array([1, 1, 1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]



    def step(self, action):
        action = np.clip(action, np.array([0., 0., 0., 0.]), np.array([1, 1, 1, 1]))
        u1, u2, u3, u4 = action

        m1, m2, m3, m4, p1, p2, p3, p4 = self.state
        m1_dot = self.c1 / (self.K + np.square(p4)) - self.c2 * m1 + self.b1 * u1
        p1_dot = self.c3 * m1 - self.c4*p1

        m2_dot = self.c1 / (self.K + np.square(p1)) - self.c2 * m2 + self.b2 * u2
        p2_dot = self.c3 * m2 - self.c4 * p2

        m3_dot = self.c1 / (self.K + np.square(p2)) - self.c2 * m3 + self.b3 * u3
        p3_dot = self.c3 * m3 - self.c4 * p3

        m4_dot = self.c1 / (self.K + np.square(p3)) - self.c2 * m4 + self.b4 * u4
        p4_dot = self.c3 * m4 - self.c4 * p4

        m1 = np.max([m1 + m1_dot * self.dt + np.random.uniform(-self.sigma,self.sigma,1),np.zeros([1])])
        m2 = np.max([m2 + m2_dot * self.dt + np.random.uniform(-self.sigma,self.sigma,1),np.zeros([1])])
        m3 = np.max([m3 + m3_dot * self.dt + np.random.uniform(-self.sigma,self.sigma,1),np.zeros([1])])
        m4 = np.max([m4 + m4_dot * self.dt + np.random.uniform(-self.sigma, self.sigma, 1), np.zeros([1])])

        p1 = np.max([p1 + p1_dot * self.dt + np.random.uniform(-self.sigma,self.sigma,1),np.zeros([1])])
        p2 = np.max([p2 + p2_dot * self.dt + np.random.uniform(-self.sigma,self.sigma,1),np.zeros([1])])
        p3 = np.max([p3 + p3_dot * self.dt + np.random.uniform(-self.sigma,self.sigma,1),np.zeros([1])])
        p4 = np.max([p4 + p4_dot * self.dt + np.random.uniform(-self.sigma, self.sigma, 1), np.zeros([1])])

        self.state = np.array([m1, m2, m3, m4, p1, p2, p3, p4])
        self.t = self.t + 1
        r1, r2 = self.reference(self.t)
        cost = np.square(p1-r1)

        # if cost>100:
        #     done = True
        # else:
        #     done = False
        done = False
        return np.array([m1, m2, m3, m4, p1, p2, p3, p4, r1, p1-r1]), cost, done, dict(reference=r1,
                                                                                       state_of_interest=p1,
                                                                                       # state_of_interest=[m1, m2, m3, m4, p1, p2, p3, p4]
                                                                                       )

    def reset(self):
        self.state = self.np_random.uniform(low=0, high=5, size=(8,))
        # self.state = np.array([1,2,3,1,2,3])
        self.t = 0
        m1, m2, m3, m4, p1, p2, p3, p4 = self.state
        r1, r2 = self.reference(self.t)
        # self.state[0] = self.np_random.uniform(low=5, high=6)
        return np.array([m1, m2, m3, m4, p1, p2, p3, p4, r1, p1-r1])

    def reference(self, t):
        r1 = 8+7*np.sin((2*np.pi)*t/200)
        # r1 = 8 + 7 * np.sin((2 * np.pi) * t / 40)
        r2 = 8+7*np.sin((2*np.pi)*(t + 200/3)/200)
        return r1, r2

    def reference2(self, t):
        r1 = 8
        r2 = 8 + 7 * np.sin((2 * np.pi) * (t + 200 / 3) / 200)
        return r1, r2

    def render(self, mode='human'):

        return


if __name__=='__main__':
    env = oscillator()
    T = 600
    path = []
    t1 = []
    s = env.reset()
    for i in range(int(T/env.dt)):
        s, r, done, info = env.step(np.array([0,0, 0,0]))
        path.append(s[0:8])
        t1.append(i * env.dt)

    # path2 = []
    # t2 = []
    # env.dt = 1
    # s = env.reset()
    # for i in range(int(T / env.dt)):
    #     s, r, done, info = env.step(np.array([0, 0]))
    #     path2.append(s)
    #     t2.append(i * env.dt)
    #
    # path3 = []
    # t3 = []
    # env.dt = 0.01
    # s = env.reset()
    # for i in range(int(T / env.dt)):
    #     s, r, done, info = env.step(np.array([0, 0]))
    #     path3.append(s)
    #     t3.append(i * env.dt)
    #
    # path4 = []
    # t4 = []
    # env.dt = 0.001
    # s = env.reset()
    # for i in range(int(T / env.dt)):
    #     s, r, done, info = env.step(np.array([0, 0]))
    #     path4.append(s)
    #     t4.append(i * env.dt)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.plot(t1, path, color='blue', label='0.1')
    # ax.plot(t2, path2, color='red',label='1')
    #
    # ax.plot(t3, path3, color='black', label='0.01')
    # ax.plot(t4, path4, color='orange', label='0.001')
    handles, labels = ax.get_legend_handles_labels()

    ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
    plt.show()
    print('done')




