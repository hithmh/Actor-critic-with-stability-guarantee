


"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""


import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from .PCCS_plant.Models import PCCPlant
import scipy.io as sio

class pccs_env(gym.Env):

    def __init__(self):
        self.t = 0
        self.delay = 1
        self.action_sample_period = np.random.randint(10, 30)
        self.p_sampling_period = 20

        Nx = 103  # Number of differential states
        Nz = 7  # Number of algebraic states
        Nd = 2  # Number of disturbances
        Nu = 3  # Number of inputs
        Np = 1  # Number of parameters
        Ny = 2  # Number of outputs

        # self.observed_dims = [30, 80]
        self.observed_dims = range(Nx)
        # self.observed_dims = [
        #     20, 21, 22, 23, 24,
        #     45, 46, 47, 48, 49,
        #     70, 71, 72, 73, 74,
        #     95, 96, 97, 98, 99,
        #     100, 101, 102,
        # ]
        self.observed_ouput_dims = [1]
        self.observed_state_dims = [
            20,
            70,
        ]
        # self.output_dim = len(self.observed_ouput_dims) + len(self.observed_state_dims)
        self.output_dim = len(self.observed_ouput_dims)
        self.z_dim = Nz

        self.dt = dt = 4000 * 1e-2  # Plant simulation time step or sampling time, s # must be same as the sampling time in the data generation
        tf = self.dt * 2000  # Total simulation time, s # over-ridden
        eps = 1E-30  # A very small number
        self.Nt = int(round(tf / dt))  # Number of simulation points # over-ridden


        # self.xs = np.array([0.1763, 0.6731, 480.3165, 0.1965, 0.6536, 472.7863, 0.0651, 0.6703, 474.8877])


        self.us = sio.loadmat('./envs/PCCS_plant/U6_sets.mat')['U']
        self.us = self.us.T
        self.z0 = np.load('./envs/PCCS_plant/z0.npy')
        self.x0 = np.load('./envs/PCCS_plant/x0.npy')

        self.create_simulator()


        # self.wwtp_sim = mpc.DiscreteSimulator(ode_bsm1model, Delta, [Nx, Nu], ["x", "u"])

        high = np.array(np.ones(len(self.observed_dims)))

        # self.action_low = np.array([0.02, 0.194, 0.02])
        # self.action_high = np.array([0.04, 0.333, 0.04])

        self.action_low = np.array([20, 700, 20], dtype=np.float64)
        self.action_high = np.array([40., 1200, 40], dtype=np.float64)

        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float32)
        self.p_space = spaces.Box(low=np.array([0.2]), high=np.array([1.]), dtype=np.float64)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        # self.ps = sio.loadmat('./envs/PCCS_plant/P6_sets.mat')['P']  # the disturbance is engine load ratio. The value should eb between  0.1 to 1.
        self.ps = self.get_ps(self.Nt)
        self.seed()
        self.viewer = None
        self.state = None
        self.state_buffer = state_buffer(self.delay)

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def create_simulator(self):
        Nx = 103  # Number of differential states
        Nz = 7  # Number of algebraic states
        Nd = 2  # Number of disturbances
        Nu = 3  # Number of inputs
        Np = 1  # Number of parameters
        Ny = 2  # Number of outputs
        pccplant = PCCPlant.PCCPlant()

        xd = SX.sym('x', Nx)  # Differential states
        xz = SX.sym('z', Nz)  # Algebraic states
        u = SX.sym('u', Nu)  # Control inputs
        p = SX.sym('p', Np)  # Parameters. Currently fixed in the model so unused.

        ode = pccplant.getODE(xd, xz, u, p)
        alg = pccplant.getALG(xd, xz)
        dae = {'x': xd, 'z': xz, 'p': vertcat(u, p), 'ode': ode, 'alg': alg}
        opts = {"tf": self.dt, "abstol": 1e-5}  # interval length
        self.plant_sim = integrator('I', 'idas', dae, opts)

    def step(self, action, impulse = 0):

        # def metricCalc(x, z, u):
        #     # X = SX.zeros(2)
        #     # waste_heat = 47*p[0]* 1.12 * (360-150) * 0.748 / 1000
        #     steam_heat = (u[1] * 42200 / 2762.83 * (2762.83 - 697.5) / 3600 / 1000 * 1000 * 1E-6)
        #     X1 = ((0.00223 - x[30]) / 0.00223) * 100
        #     X2 = (steam_heat) / ((939.7581 * u[0] * 0.001 * z[4] * x[80] * 44.01 * 0.001))
        #     return np.array([X1, X2])

        action = np.clip(action, self.action_low, self.action_high, dtype=np.float64)
        # input = np.concatenate((action, self.Q0[self.t], self.Z0[self.t]))


        sol = self.plant_sim(x0=self.state, z0=self.z, p=vertcat(action, self.p))
        self.state = sol["xf"].full()[:, 0]
        self.z = sol["zf"].full()[:, 0]
        last_d = self.p
        y = self.measurements(self.state, self.p)

        cost = self.calc_cost(y, action)

        self.state_buffer.memorize(self.state)
        self.t += 1
        s = self.observe()
        self.p = self.get_p()


        done = False
        data_collection_done = False

        # if self.t == self.ps.shape[1]:
        # if self.t == self.Nt:
        #     done = True
        #     data_collection_done = True

        
        return s, cost, done, dict(data_collection_done=data_collection_done, output=self.output(y, self.state), d=last_d, z=self.z)

    def observe(self):
        '''
        A function that picks the observed states from the state vector. The dimensions to be observed are specified by self.observed_dims.
        Returns:

        '''
        x = self.state
        y = np.zeros(len(self.observed_dims))
        for i in range(len(self.observed_dims)):
            y[i] = x[self.observed_dims[i]]

        return y

    def output(self, y, s):
        output_y = np.zeros(len(self.observed_ouput_dims))
        output_x = np.zeros(len(self.observed_state_dims))
        for i in range(len(self.observed_ouput_dims)):
            output_y[i] = y[self.observed_ouput_dims[i]]
        for i in range(len(self.observed_state_dims)):
            output_x[i] = s[self.observed_state_dims[i]]
        output = np.concatenate((output_y, output_x))
        return output_y

    def measurements(self, x, p):
        engine_cap = 10800 * 2  # kW, times 2 because there are 2 main engines as per Luo and Wang 2017 paper

        SFOC = 177.5  # g/kWh
        air_intake = 6.27  # kg/kWh

        # Flue gas flow rate components
        CO2_flow = (SFOC * engine_cap * p / 1000000) * 86.66 / 100 * 44 / 12  # %C = 86.6, MW C = 12, MW CO2 = 44, unit is ton/h
        H2O_flow = ( SFOC * engine_cap * p / 1000000) * 12.29 / 100 * 18 / 2  # %H2 = 12.29, MW H2 = 2, MW H2O = 18, unit is ton/h
        N2_flow = air_intake * engine_cap * p * 92.5 / 100 / 1000  # %N2 = 92.5, MW N = 12, MW CO2 = 44, unit is ton/h
        total_flue_gas = (CO2_flow + H2O_flow + N2_flow) * 1000 / 0.7 / 3600  # convert unit from ton/h to m3/s, density of flue gas average is 0.7 kg/m3

        co2_release = x[30] * 44.01 * total_flue_gas * 3600  # unabosrbed CO2 (kg/hr)
        reb_tem = x[102]  # reboiler temperature (K) if unit is (C), x[102]-273.15
        return np.array([co2_release, reb_tem], dtype=np.float64)


    def calc_cost(self, y, u):
        alpha = 0.05
        beta = 1.2852
        y_limit = 0.5
        cost = alpha * np.clip(y[0]-y_limit, 0, np.inf)/3600 #+ beta * u[1]/3600

        return cost


    def reset(self):

        self.state_buffer.reset()
        self.t = 0
        self.time = 0
        self.a_holder = self.action_space.sample()
        self.p_holder = self.p_space.sample()
        self.state = self.x0
        self.z = self.z0

        # self.create_simulator()
        self.p = self.get_p()

        # d = np.concatenate((self.Q0[self.t], self.Z0[self.t]))
        # self.t += 1
        # action = self.action_space.sample()
        # for i in range(self.sampling_steps):
        #     # process_noise = np.random.normal(np.zeros_like(self.kw), self.kw)
        #     # process_noise = np.clip(process_noise, -self.bw, self.bw)
        #     self.state = self.state + ode_bsm1model(self.state, action, d) * self.h #+ process_noise * self.h
        # index = random.randint(0, len(self.data)-1)
        # index = 2
        # Q0 = self.data[index][14]
        # Z0 = self.data[index][1:14]
        # Nsim = Q0.shape[0]
        # self.p = np.zeros((self.Np, Nsim))
        # self.p[0, :] = Q0
        # self.p[1:14, :] = Z0
        s = self.observe()
        # s, _, _, _ = self.step(self.get_action())
        return s


    
    def render(self, mode='human'):

        return


    def get_action(self):

        if self.t % self.action_sample_period == 0:
            self.a_holder = self.action_space.sample()
            self.action_sample_period = np.random.randint(10, 30)
        a = self.a_holder + np.random.normal(np.zeros_like(self.action_high), (self.a_holder)*0.05)
        a = np.clip(a, self.action_low, self.action_high)

        # a = self.us[self.t]

        return a

    def get_p(self):

        # if self.t % self.p_sampling_period == 0:
        #     self.p_holder = np.clip(self.p_space.sample(), self.p_holder-0.2, self.p_holder+0.2)
        # p = self.p_holder + np.random.normal(np.zeros_like(self.p_holder), 0.05)
        # p = np.clip(p, 0.1, 1)
        # p = self.p_holder
        p = self.ps[self.t]

        return p

    def get_ps(self, length):
        ps = []
        p_holder = self.p_space.sample()
        for t in range(length):
            if t % self.p_sampling_period == 0:
                p_holder = np.clip(self.p_space.sample(), p_holder - 0.2, p_holder + 0.2)
            ps.append(p_holder + np.random.normal(np.zeros_like(p_holder), 0.01))
            # ps.append(self.p_space.sample())

        return np.array(ps)

    def observe_d_sequence(self, horizon):
        t = self.t
        d_sequence = []
        for i in range(horizon):
            d_sequence.append(self.ps[t+i])
        return np.array(d_sequence)
class state_buffer(object):

    def __init__(self, delay):

        self.delay = delay
        self.memory = []


    def memorize(self, s):

        self.memory.append(s)
        return

    def get_state(self, t):

        if t < self.delay:
            return None
        else:
            return self.memory[t]

    def reset(self):
        self.memory = []



def main():
    env = pccs_env()
    T = 1344
    cost = 0
    path = []
    a_path = []
    t1 = []
    s = env.reset()
    for i in range(int(T)):
        action = env.get_action()
        s, r, done, info = env.step(action)
        path.append(s)
        cost += r
        a_path.append(action)
        t1.append(i)
    path = np.array(path)
    state_dim = 2
    fig, ax = plt.subplots(state_dim, sharex=True, figsize=(15, 15))
    t = range(T)
    for i in range(state_dim):
        ax[i].plot(t, path[:, i], color='red')
    # fig = plt.figure(figsize=(9, 6))
    # ax = fig.add_subplot(111)
    # ax.plot(t1, path)
    # # ax.plot(t2, path2, color='red',label='1')
    # #
    # # ax.plot(t3, path3, color='black', label='0.01')
    # # ax.plot(t4, path4, color='orange', label='0.001')
    # handles, labels = ax.get_legend_handles_labels()
    #
    # ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
    # plt.show()
    # print('done')

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.plot(t1, a_path)
    # ax.plot(t2, path2, color='red',label='1')
    #
    # ax.plot(t3, path3, color='black', label='0.01')
    # ax.plot(t4, path4, color='orange', label='0.001')
    handles, labels = ax.get_legend_handles_labels()

    ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
    plt.show()
    print('done')

if __name__=='__main__':
    main()










