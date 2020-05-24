import pybullet_envs.bullet.minitaur_gym_env as e
import numpy as np
import time
from pybullet_envs.bullet import minitaur
import math
from gym import spaces

OBSERVATION_EPS = 0.01

class minitaur_env(e.MinitaurBulletEnv):
    def __init__(self, **kwargs):
        self._velocity_weight = 0.
        self.target_velocity = 1.
        self.target_position_range = np.asarray([1, 1])
        self.target_velocity_range = np.asarray(1)
        super(minitaur_env, self).__init__(**kwargs)


        self._distance_weight = 1.
        self._energy_weight = 0.
        self._drift_weight = 0
        self._shake_weight = 0


        max_forward_speed = [np.inf] # m/s
        max_position = [np.inf, np.inf]

        self.target_position = self.target_position_range

        # if self._distance_weight>0:
        #
        #     observation_high = (np.concatenate([self.minitaur.GetObservationUpperBound(), max_position, max_position, max_position]) + OBSERVATION_EPS)
        #     observation_low = (
        #                 -np.concatenate([self.minitaur.GetObservationUpperBound(), max_position, max_position, max_position]) - OBSERVATION_EPS)
        #     self.observation_space = spaces.Box(observation_low, observation_high, dtype=np.float32)

        if self._velocity_weight> 0:
            observation_high = (np.concatenate([self.minitaur.GetObservationUpperBound(), max_forward_speed, max_forward_speed, max_forward_speed]) + OBSERVATION_EPS)
            observation_low = (-np.concatenate(
            [self.minitaur.GetObservationUpperBound(), max_forward_speed, max_forward_speed,
             max_forward_speed]) - OBSERVATION_EPS)

            self.observation_space = spaces.Box(observation_low, observation_high, dtype=np.float32)

    def _reward(self):
        current_base_position = self.minitaur.GetBasePosition()
        forward_distance = current_base_position[0] - self._last_base_position[0]
        forward_speed = forward_distance / self._time_step
        forward_reward = (self.target_velocity - forward_speed)**2
        distance_reward = (current_base_position[0] - self.target_position[0])**2 + (current_base_position[1] - self.target_position[1])**2
        drift_reward = (current_base_position[1] - self._last_base_position[1])**2
        shake_reward = -abs(current_base_position[2] - self._last_base_position[2])
        self._last_base_position = current_base_position
        energy_reward = np.abs(
            np.dot(self.minitaur.GetMotorTorques(),
                   self.minitaur.GetMotorVelocities())) * self._time_step
        reward = (self._velocity_weight * forward_reward + self._energy_weight * energy_reward +
                  self._drift_weight * drift_reward + self._shake_weight * shake_reward + self._distance_weight * distance_reward)
        self._objectives.append([forward_reward, energy_reward, drift_reward, shake_reward, distance_reward])
        return reward

    def _get_observation(self):
        current_base_position = self.minitaur.GetBasePosition()
        extended_obs = []
        # if self._distance_weight> 0:
        #     extended_obs = np.concatenate([current_base_position[0:2], self.target_position,  self.target_position - current_base_position[0:2]])
        if self._velocity_weight>0:
            forward_distance = current_base_position[0] - self._last_base_position[0]
            forward_speed = forward_distance / self._time_step
            extended_obs = np.asarray([forward_speed, self.target_velocity, self.target_velocity - forward_speed])

        self._observation = np.concatenate([self.minitaur.GetObservation(), extended_obs])
        return self._observation

    def reset(self):
        if self._hard_reset:
            self._pybullet_client.resetSimulation()
            self._pybullet_client.setPhysicsEngineParameter(
                numSolverIterations=int(self._num_bullet_solver_iterations))
            self._pybullet_client.setTimeStep(self._time_step)
            plane = self._pybullet_client.loadURDF("%s/plane.urdf" % self._urdf_root)
            self._pybullet_client.changeVisualShape(plane, -1, rgbaColor=[1, 1, 1, 0.9])
            self._pybullet_client.configureDebugVisualizer(
                self._pybullet_client.COV_ENABLE_PLANAR_REFLECTION, 0)
            self._pybullet_client.setGravity(0, 0, -10)
            acc_motor = self._accurate_motor_model_enabled
            motor_protect = self._motor_overheat_protection
            self.minitaur = (minitaur.Minitaur(pybullet_client=self._pybullet_client,
                                               urdf_root=self._urdf_root,
                                               time_step=self._time_step,
                                               self_collision_enabled=self._self_collision_enabled,
                                               motor_velocity_limit=self._motor_velocity_limit,
                                               pd_control_enabled=self._pd_control_enabled,
                                               accurate_motor_model_enabled=acc_motor,
                                               motor_kp=self._motor_kp,
                                               motor_kd=self._motor_kd,
                                               torque_control_enabled=self._torque_control_enabled,
                                               motor_overheat_protection=motor_protect,
                                               on_rack=self._on_rack,
                                               kd_for_pd_controllers=self._kd_for_pd_controllers))
        else:
            self.minitaur.Reset(reload_urdf=False)

        if self._env_randomizer is not None:
            self._env_randomizer.randomize_env(self)

        self._env_step_counter = 0
        self._last_base_position = [0, 0, 0]
        self._objectives = []
        self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw,
                                                         self._cam_pitch, [0, 0, 0])
        if not self._torque_control_enabled:
            for _ in range(100):
                if self._pd_control_enabled or self._accurate_motor_model_enabled:
                    self.minitaur.ApplyAction([math.pi / 2] * 8)
                self._pybullet_client.stepSimulation()
        if self._velocity_weight > 0:
            self.target_velocity = np.random.uniform(0, self.target_velocity_range)

        # if self._distance_weight > 0:
        #     self.target_position = np.random.uniform(-self.target_position_range, self.target_position_range)

        return self._noisy_observation()

    def step(self, action):
        """Step forward the simulation, given the action.

        Args:
          action: A list of desired motor angles for eight motors.

        Returns:
          observations: The angles, velocities and torques of all motors.
          reward: The reward for the current state-action pair.
          done: Whether the episode has ended.
          info: A dictionary that stores diagnostic information.

        Raises:
          ValueError: The action dimension is not the same as the number of motors.
          ValueError: The magnitude of actions is out of bounds.
        """
        if self._is_render:
            # Sleep, otherwise the computation takes less time than real time,
            # which will make the visualization like a fast-forward video.
            time_spent = time.time() - self._last_frame_time
            self._last_frame_time = time.time()
            time_to_sleep = self._action_repeat * self._time_step - time_spent
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
            base_pos = self.minitaur.GetBasePosition()
            camInfo = self._pybullet_client.getDebugVisualizerCamera()
            curTargetPos = camInfo[11]
            distance = camInfo[10]
            yaw = camInfo[8]
            pitch = camInfo[9]
            targetPos = [
                0.95 * curTargetPos[0] + 0.05 * base_pos[0], 0.95 * curTargetPos[1] + 0.05 * base_pos[1],
                curTargetPos[2]
            ]

            self._pybullet_client.resetDebugVisualizerCamera(distance, yaw, pitch, base_pos)
        action = self._transform_action_to_motor_command(action)
        for _ in range(self._action_repeat):
            self.minitaur.ApplyAction(action)
            self._pybullet_client.stepSimulation()

        self._env_step_counter += 1
        reward = self._reward()
        done = self._termination(reward)
        if done:
            reward = reward + (500.-self._env_step_counter)
        return np.array(self._noisy_observation()), reward, done, {}

    def _termination(self, reward):
        # return self.is_fallen()
        # done = False
        # current_base_position = self.minitaur.GetBasePosition()
        # forward_distance = current_base_position[0] - self._last_base_position[0]
        # forward_speed = forward_distance / self._time_step
        # if forward_speed<-0.1 or self.is_fallen():
        #     done = True

        # return self.is_fallen()

        return False

    def is_fallen(self):
        """Decide whether the minitaur has fallen.

        If the up directions between the base and the world is larger (the dot
        product is smaller than 0.85) or the base is very low on the ground
        (the height is smaller than 0.13 meter), the minitaur is considered fallen.

        Returns:
          Boolean value that indicates whether the minitaur has fallen.
        """
        orientation = self.minitaur.GetBaseOrientation()
        rot_mat = self._pybullet_client.getMatrixFromQuaternion(orientation)
        local_up = rot_mat[6:]
        pos = self.minitaur.GetBasePosition()
        return (np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < 0.85)
