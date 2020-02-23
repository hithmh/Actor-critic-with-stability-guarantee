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

class CartPoleEnv_cost(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24°           24°
        3	Pole Velocity At Tip      -Inf            Inf

    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right

        Note: The amount the velocity is reduced or increased is not fixed as it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value between ±0.05

    Episode Termination:
        Pole Angle is more than ±12°
        Cart Position is more than ±2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.gravity = 10
        # 1 0.1 0.5 original
        self.masscart = 1
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 200
        self.tau = 0.005  # seconds between state updates
        self.kinematics_integrator = 'euler'
        self.cons_pos = 4
        self.target_pos = 0
        # Angle at which to fail the episode
        self.theta_threshold_radians = 20 * 2 * math.pi / 360
        # self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 10
        # self.max_v=1.5
        # self.max_w=1
        # FOR DATA
        self.max_v = 50
        self.max_w = 50
        # self.A = np.array([[1, 0.0200000000000000, -0.000146262956164120, -9.75295709398121e-07],
        #                    [0, 1, -0.0146184465178444, -0.000146262956164120],
        #                    [0, 0, 0.996782214976383, 0.0199785434944732],
        #                    [0, 0, -0.321605822193865, 0.996782214976383]])
        # self.B = np.array([[0.000195114814924033],
        #                    [0.0195107679379903],
        #                    [0.000292525910329313],
        #                    [0.0292368928359034]])
        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            self.max_v,
            self.theta_threshold_radians * 2,
            self.max_w])

        self.action_space = spaces.Box(low=-self.force_mag, high=self.force_mag, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_params(self, length, mass_of_cart, mass_of_pole, gravity):
        self.gravity = gravity
        self.length = length
        self.masspole = mass_of_pole
        self.masscart = mass_of_cart
        self.total_mass = (self.masspole + self.masscart)
        self.polemass_length = (self.masspole * self.length)

    def get_params(self):

        return self.length, self.masspole, self.masscart, self.gravity

    def reset_params(self):

        self.gravity = 10
        self.masscart = 1
        self.masspole = 0.1
        self.length = 0.5
        self.total_mass = (self.masspole + self.masscart)
        self.polemass_length = (self.masspole * self.length)

    def step(self, action, impulse=0, process_noise=np.zeros([5])):
        a = 0
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # self.gravity = np.random.normal(10, 2)
        # self.masscart = np.random.normal(1, 0.2)
        # self.masspole = np.random.normal(0.1, 0.02)
        self.total_mass = (self.masspole + self.masscart)
        state = self.state

        x, x_dot, theta, theta_dot = state
        force = np.random.normal(action, 0)# wind
        force = force + process_noise[0] + impulse
        # force = action
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot+ process_noise[2]
            x_dot = x_dot + self.tau * xacc + process_noise[4]
            # x_dot = np.clip(x_dot, -self.max_v, self.max_v)
            theta = theta + self.tau * theta_dot + process_noise[1]
            theta_dot = theta_dot + self.tau * thetaacc + process_noise[3]

            # theta_dot = np.clip(theta_dot, -self.max_w, self.max_w)
        elif self.kinematics_integrator == 'friction':
            xacc = -0.1 * x_dot / self.total_mass + temp - self.polemass_length * thetaacc * costheta / self.total_mass
            x = x + self.tau * x_dot + process_noise[2]
            x_dot = x_dot + self.tau * xacc + process_noise[4]
            # x_dot = np.clip(x_dot, -self.max_v, self.max_v)
            theta = theta + self.tau * theta_dot + process_noise[1]
            theta_dot = theta_dot + self.tau * thetaacc+ process_noise[3]
            # theta_dot = np.clip(theta_dot, -self.max_w, self.max_w):
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc + process_noise[4]
            x = x + self.tau * x_dot  + process_noise[2]
            theta_dot = theta_dot + self.tau * thetaacc+ process_noise[3]
            theta = theta + self.tau * theta_dot + process_noise[1]
        self.state = np.array([x, x_dot[0], theta, theta_dot[0]])
        done = abs(x) > self.x_threshold \
               or theta < -self.theta_threshold_radians \
               or theta > self.theta_threshold_radians
        done = bool(done)
        if x < -self.x_threshold \
                or x > self.x_threshold:
            a = 1
        r1 = ((self.x_threshold/10 - abs(x-self.target_pos))) / (self.x_threshold/10)  # -4-----1
        r2 = ((self.theta_threshold_radians / 4) - abs(theta)) / (self.theta_threshold_radians / 4)  # -3--------1
        # r1 = max(10 * (1 - ((x-self.target_pos)/self.x_threshold) **2), 1)
        # r2 = max(10 * (1 - np.abs((theta)/self.theta_threshold_radians)), 1)
        # cost1=(self.x_threshold - abs(x))/self.x_threshold
        e1 = (abs(x)) / self.x_threshold
        e2 = (abs(theta)) / self.theta_threshold_radians
        cost = COST_V1(r1, r2, e1, e2, x, x_dot, theta, theta_dot)
        # cost = 0.1+10*max(0, (self.theta_threshold_radians - abs(theta))/self.theta_threshold_radians) \
        #     #+ 5*max(0, (self.x_threshold - abs(x-self.target_pos))/self.x_threshold)\
        cost = 1* x**2/100 + 20 *(theta/ self.theta_threshold_radians)**2
        l_rewards = 0
        if abs(x)>self.cons_pos:
            violation_of_constraint = 1
        else:
            violation_of_constraint = 0


        # ## linear update
        # self.linear_state = self.A @ self.linear_state + self.B @ action
        # x = self.linear_state[0]
        # theta = self.linear_state[2]
        # done = abs(x) > self.x_threshold \
        #        or theta < -self.theta_threshold_radians \
        #        or theta > self.theta_threshold_radians
        # done = bool(done)
        # if x < -self.x_threshold \
        #         or x > self.x_threshold:
        #     a = 1

        return self.state, cost, done, dict(hit=a,
                                            l_rewards=l_rewards,
                                            cons_pos=self.cons_pos,
                                            cons_theta=self.theta_threshold_radians,
                                            target=self.target_pos,
                                            violation_of_constraint=violation_of_constraint
                                            )

    def reset(self):
        self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(4,))
        # self.state[0] = self.np_random.uniform(low=5, high=6)
        self.state[0] = self.np_random.uniform(low=-5, high=5)
        self.steps_beyond_done = None
        # self.linear_state = np.array(self.state)
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 800
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            # Render the target position
            self.target = rendering.Line((self.target_pos * scale + screen_width / 2.0, 0),
                                         (self.target_pos * scale + screen_width / 2.0, screen_height))
            self.target.set_color(1, 0, 0)
            self.viewer.add_geom(self.target)


            # # Render the constrain position
            # self.cons = rendering.Line((self.cons_pos * scale + screen_width / 2.0, 0),
            #                              (self.cons_pos * scale + screen_width / 2.0, screen_height))
            # self.cons.set_color(0, 0, 1)
            # self.viewer.add_geom(self.cons)

        if self.state is None: return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def COST_1000(r1, r2, e1, e2, x, x_dot, theta, theta_dot):
    cost = np.sign(r2) * ((10 * r2) ** 2) - 4 * abs(x) ** 2
    return cost

def COST_V3(r1, r2, e1, e2, x, x_dot, theta, theta_dot):
    cost = np.sign(r2) * ((10 * r2) ** 2) - abs(x) ** 4
    return cost

def COST_V1(r1, r2, e1, e2, x, x_dot, theta, theta_dot):
    cost = 20*np.sign(r2) * ((r2) ** 2)+ 1* np.sign(r1) * (( r1) ** 2)
    return cost


def COST_V2(r1, r2, e1, e2, x, x_dot, theta, theta_dot):
    cost = 5 * max(r2, 0) + 1* max(r1,0) + 1
    return cost
