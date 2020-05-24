from gym.envs.mujoco.swimmer import SwimmerEnv
import numpy as np

class swimmer_env(SwimmerEnv):
    def __init__(self):
        self.target_vel = .3
        super(swimmer_env, self).__init__()


    def step(self, a):
        ctrl_cost_coeff = 0  # 0.0001
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        vel = (xposafter - xposbefore) / self.dt
        reward_fwd = abs( vel - self.target_vel)
        reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

