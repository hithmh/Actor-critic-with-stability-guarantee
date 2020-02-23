import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class HalfCheetahEnv_cost(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.des_v = 1.
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)



    def step(self, action, process_noise = np.zeros([23])):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action + process_noise[0:6], self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        cost_ctrl = 0.1 * np.square(action).sum()
        v = (xposafter - xposbefore)/self.dt
        run_cost = np.square(v-self.des_v)
        # reward_run = xposafter
        reward = run_cost #+ cost_ctrl
        if abs(ob[2]) > np.pi/2:
            done = True
        else:
            done = False


        l_rewards = 0.

        if abs(run_cost)>3:
            violation_of_constraint = 1
        else:
            violation_of_constraint = 0

        # print(xposafter)
        return ob, reward, done, dict(reference=self.des_v, state_of_interest=v)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
