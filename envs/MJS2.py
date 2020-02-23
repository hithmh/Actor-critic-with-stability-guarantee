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

class MJS(gym.Env):

    def __init__(self):

        self.A = []
        self.B = []
        rho = 0.859
        self.A.append(rho * np.array([
            [-0.4227, 0.7710],
            [-1.1600, -0.6912]]))
        self.A.append(rho * np.array([
            [-0.5084, 0.4536],
            [1.0901, -0.7266]]))
        self.A.append(rho * np.array([
            [-0.4772, 0.7313],
            [1.3938, -0.7266]]))

        self.B.append(np.array([
            [1],
            [2]]))
        self.B.append(np.array([
            [0],
            [0]]))
        self.B.append(np.array([
            [1],
            [2]]))

        self.sigma = 1
        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([1, 1, 1])

        self.action_space = spaces.Box(low=np.array([0.]), high=np.array([1,]), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]



    def step(self, action):
        action = np.clip(action, -np.array([1,]), np.array([1,]))


        self.state = self.A[self.sigma].dot(self.state) + self.B[self.sigma].dot(action)
        self.t = self.t + 1

        s1, s2 = self.state
        self.sigma = min(int(np.floor(np.random.uniform(low=0,high=3, size=(1,)))[0]), len(self.A))
        cost = self.state.dot(self.state)
        if abs(s1)>100 or abs(s2)>100:
            done = True
        else:
            done = False
        return np.array([s1, s2, self.sigma]), cost, done, dict(reference=0, state_of_interest=[s1, s2])

    def reset(self):
        self.state = self.np_random.uniform(low=-1, high=1, size=(2,))
        # self.state = np.array([1,2,3,1,2,3])
        self.t = 0
        s1, s2 = self.state
        self.sigma = min(int(np.floor(np.random.uniform(low=0,high=3, size=(1,)))[0]), len(self.A))

        # self.state[0] = self.np_random.uniform(low=5, high=6)
        return np.array([s1, s2, self.sigma])



    def render(self, mode='human'):

        return


if __name__=='__main__':
    env = MJS()
    T = 400
    path = []
    t1 = []
    s = env.reset()
    for i in range(int(T)):
        s, r, done, info = env.step(np.array([0]))
        path.append(info['state_of_interest'])
        t1.append(i)

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