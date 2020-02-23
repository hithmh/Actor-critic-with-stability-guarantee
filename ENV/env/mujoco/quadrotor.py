import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class QuadrotorEnv(gym.Env):
    """
    Description:


    Source:


    Observation:
        Type: Box()
        Num	Observation                 Min         Max


    Actions:


    Reward:


    Starting State:


    Episode Termination:

    """

    def __init__(self):

        # crazyflie: physical parameters for the Crazyflie 2.0

        # This function creates a struct with the basic parameters for the
        # Crazyflie 2.0 quad rotor (without camera, but with about 5 vicon
        # markers)
        # Model assumptions based on physical measurements:
        # motor + mount + vicon marker = mass point of 3g
        # arm length of mass point: 0.046m from center
        # battery pack + main board are combined into cuboid (mass 18g) of
        # dimensions:
        # width  = 0.03m
        # depth  = 0.03m
        # height = 0.012m

        m = 0.030  # weight (in kg) with 5 vicon markers (each is about 0.25g)
        gravity = 9.81  # gravitational constant
        I = [[1.43e-5, 0, 0],
             [0, 1.43e-5, 0],
             [0, 0, 2.89e-5]]  # inertial tensor in m^2 kg
        L = 0.046  # arm length in m

        self.m = m
        self.g = gravity
        self.I = I
        self.invI = np.linalg.inv(self.I)
        self.arm_length = L

        self.max_angle = 40 * math.pi / 180  # you can specify the maximum commanded angle here
        self.max_F = 2.5 * m * self.g  # left these untouched from the nano plus
        self.min_F = 0.05 * m * self.g  # left these untouched from the nano plus

        # You can add any fields you want in self
        # for example you can add your controller gains by
        # self.k = 0, and they will be passed into controller.

        # Need to optimize
        x_bound = 10
        y_bound = 10
        z_bound = 10
        xdot_bound = 5
        ydot_bound = 5
        zdot_bound = 1
        qW_bound = 1
        qX_bound = 1
        qY_bound = 1
        qZ_bound = 0.1
        p_bound = 5
        q_bound = 1
        r_bound = 1
        high_s = np.array([
            x_bound,
            y_bound,
            z_bound,
            xdot_bound,
            ydot_bound,
            zdot_bound,
            qW_bound,
            qX_bound,
            qY_bound,
            qZ_bound,
            p_bound,
            q_bound,
            r_bound,
            x_bound,
            y_bound,
            z_bound,
        ])
        high_a = 2*np.array([0.05, 0.05, 0.02, 0.05, 0.05, 0.02])
        # high_a = np.array([0.2,0.2,0.2])
        self.action_space = spaces.Box(low=-high_a, high=high_a, dtype=np.float32)
        self.observation_space = spaces.Box(-high_s, high_s, dtype=np.float32)
        self.a_bound = self.action_space.high
        self.modify_action_scale = True
        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.desire_state = None
        self.plot_x = []
        self.plot_y = []
        self.plot_z = []
        self.t_x = []
        self.t_y = []
        self.t_z = []
        self.thre = 1
        self.fig_n = 0
        self.t = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def quadEOM(self, t, s, F, M):
        # self.m =np.random.normal(self.m, 0.2*self.m)
        # self.g =np.random.normal(self.g, 0.1*self.g)

        # QUADEOM Solve quadrotor equation of motion
        # quadEOM calculate the derivative of the state vector
        # INPUTS:
        # t      - 1 x 1, time
        # s      - 13 x 1, state vector = [x, y, z, xd, yd, zd, qw, qx, qy, qz, p, q, r]
        # F      - 1 x 1, thrust output from controller (only used in simulation)
        # M      - 3 x 1, moments output from controller (only used in simulation)
        # self   -  output from init() and whatever parameters you want to pass in
        #  OUTPUTS:
        # sdot   - 13 x 1, derivative of state vector s
        self.A = [[0.25, 0, -0.5 / self.arm_length],
                  [0.25, 0.5 / self.arm_length, 0],
                  [0.25, 0, 0.5 / self.arm_length],
                  [0.25, -0.5 / self.arm_length, 0]]

        prop_thrusts = np.dot(self.A, [F, M[0], M[1]])

        prop_thrusts_clamped = np.maximum(np.minimum(prop_thrusts, self.max_F / 4), self.min_F / 4)
        B = [[1, 1, 1, 1],
             [0, self.arm_length, 0, -self.arm_length],
             [-self.arm_length, 0, self.arm_length, 0]]

        F = np.dot(B[0], prop_thrusts_clamped)

        M = np.reshape([np.dot(B[1:3], prop_thrusts_clamped)[0], np.dot(B[1:3], prop_thrusts_clamped)[1], M[2]], [3])
        # print("real F", F,"real M", M)
        # Assign states
        x = s[0]
        y = s[1]
        z = s[2]
        xdot = s[3]
        ydot = s[4]
        zdot = s[5]
        qW = s[6]
        qX = s[7]
        qY = s[8]
        qZ = s[9]
        p = s[10]
        q = s[11]
        r = s[12]
        quat = [qW, qX, qY, qZ]

        bRw = QuatToRot(quat)
        wRb = bRw.T

        # Acceleration
        accel = 1 / self.m * (np.dot(wRb, [[0], [0], [F]]) - [[0], [0], [self.m * self.g]])
        # Angular velocity
        K_quat = 2  # this enforces the magnitude 1 constraint for the quaternion
        quaterror = 1 - (qW * qW + qX * qX + qY * qY + qZ * qZ)
        qdot = np.dot(np.multiply([[0., -p, -q, -r],
                                   [p, 0., -r, q],
                                   [q, r, 0., -p],
                                   [r, -q, p, 0.]], -1 / 2), quat) + np.multiply(K_quat * quaterror, quat)

        # Angular acceleration
        omega = [p, q, r]
        pqrdot = np.dot(self.invI, (M - np.cross(omega, np.dot(self.I, omega))))

        # Assemble sdot
        sdot = np.zeros([13])
        sdot[0] = xdot
        sdot[1] = ydot
        sdot[2] = zdot
        sdot[3] = accel[0]
        sdot[4] = accel[1]
        sdot[5] = accel[2]
        sdot[6] = qdot[0]
        sdot[7] = qdot[1]
        sdot[8] = qdot[2]
        sdot[9] = qdot[3]
        sdot[10] = pqrdot[0]
        sdot[11] = pqrdot[1]
        sdot[12] = pqrdot[2]
        return sdot

    def controller(self, desired_state, x):
        # self.m =np.random.normal(0.030, 0.003)
        # self.g =np.random.normal(9.81, 0.3)
        # CONTROLLER quadrotor controller
        # The current states are:
        # pos, vel, euler = [roll;pitch;yaw], qd{qn}.omega
        # The desired states are:
        # pos_des, vel_des, acc_des, yaw_des, yawdot_des
        # Using these current and desired states, you have to compute the desired controls% position controller params

        # position controller params
        Kp = [15, 15, 30]
        Kd = [12, 12, 10]

        # attitude controller params
        KpM = np.ones([3]) * 3000
        KdM = np.ones([3]) * 300

        # desired_state=[pos,vel,acc,yaw,yawdot]
        # x y z xdot ydot zdot qw qx qy qz p q r
        [pos, vel, euler, omega] = stateToQd(x)
        pos_des, vel_des, acc_des, yaw_des, yawdot_des = desired_state[0], desired_state[1], desired_state[2], \
                                                         desired_state[3], desired_state[4]

        acc_des = acc_des + Kd * (np.subtract(vel_des, vel)) + Kp * (np.subtract(pos_des, pos))
        #
        #  Desired roll, pitch and yaw
        phi_des = 1 / self.g * (acc_des[0] * np.sin(yaw_des) - acc_des[1] * np.cos(yaw_des))
        theta_des = 1 / self.g * (acc_des[0] * np.cos(yaw_des) + acc_des[1] * np.sin(yaw_des))
        psi_des = yaw_des
        #
        euler_des = [phi_des, theta_des, psi_des]
        pqr_des = [0, 0, yawdot_des]

        # Thurst
        # qd{qn}.acc_des(3);
        F = self.m * (self.g + acc_des[2])
        # Moment

        M_ = np.multiply(KdM, (np.subtract(pqr_des, omega))) + np.multiply(KpM, (np.subtract(euler_des, euler)))
        M = np.dot(self.I, M_)
        return F, M

    def step(self, a):
        # For ppo, normalize the action
        if self.modify_action_scale:
            a = a / 25
            a[2] = a[2] / 2
            a[5] = a[5] / 2
        a = np.clip(a, -self.a_bound, self.a_bound)

        # desired_state = [[a[0] + self.state[0], a[1] + self.state[1], a[2] + self.state[2]],
        #                  [0, 0, 0], [0, 0, 0], 0, 0]

        desired_state = [[a[0] + self.state[0], a[1] + self.state[1], a[2] + self.state[2]],
                         [a[3] + self.state[3], a[4] + self.state[4], a[5] + self.state[5]], [0,0,0],0, 0]

        # desired_state=[pos,vel,acc,yaw,yawdot]

        F, M = self.controller(desired_state, np.array(self.state[0:13]))

        s_dot = self.quadEOM(0, self.state, F, M)
        s_ = self.state[0:13] + 0.005 * s_dot

        # s_dot= xyzdot,acc,qdot4,pqrdot3= 13维
        # 判断是否偏离太多
        target_ = trajectory('circle', self.t)
        pos = np.array([self.state[0], self.state[1], self.state[2]])
        vel = np.array([self.state[3], self.state[4], self.state[5]])
        # done =  (np.linalg.norm(pos-np.array(target[0]))+0.1*np.linalg.norm(vel-np.array(target[1])))>0.1
        # done = (np.linalg.norm(pos - np.array(target_[0]))) > 0.01
        done = (np.linalg.norm(pos - np.array(target_[0]))) > 0.1
        done = bool(done)

        # print(desired_state[0], target_[0])

        #######################################TASK###################################
        # time t's target should arrive in this step
        r = Circle_tracking(s_, self.t)

        # Update state
        self.t = self.t + 0.005
        target = trajectory('circle', self.t)
        # state= current state+ target
        self.state = np.array(
            [s_[0], s_[1], s_[2], s_[3], s_[4], s_[5], s_[6], s_[7], s_[8], s_[9], s_[10], s_[11], s_[12], target[0][0],
             target[0][1], target[0][2]])

        # Plot
        if self.fig_n % 10 == 0:
            self.plot_x.append(self.state[0])
            self.plot_y.append(self.state[1])
            self.plot_z.append(self.state[2])
            self.t_x.append(target_[0][0])
            self.t_y.append(target_[0][1])
            self.t_z.append(target_[0][2])

        self.fig_n = self.fig_n + 1
        if self.state[2]<0.3:
            l_rewards=0
        else:
            l_rewards=abs(self.state[2])*100
        violation = 1 if l_rewards>0 else 0
        return self.state, r, done, dict(l_rewards=l_rewards, violation_of_constraint=violation)

    def reset(self):
        # if self.t > self.thre:
        #     fig = plt.figure(1)
        #     ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
        #     ax.scatter(self.plot_x, self.plot_y, self.plot_z, c='r')  # 绘制数据点,颜色是红色
        #     ax.scatter(self.t_x, self.t_y, self.t_z, c='b')  # 绘制数据点,颜色是红色
        #     ax.set_zlabel('Z')  # 坐标轴
        #     ax.set_ylabel('Y')
        #     ax.set_xlabel('X')
        #     plt.draw()
        #     plt.pause(1)
            # self.thre = self.thre + 0.1
        self.fig_n = 0
        # plt.close(1)
        self.plot_x = []
        self.plot_y = []
        self.plot_z = []
        self.t_x = []
        self.t_y = []
        self.t_z = []
        self.state = np.zeros([16])
        self.state[0] = 5
        self.state[13] = 5
        self.state[1] = 0
        self.state[2] = 0
        self.state[6] = 1
        self.t = 0
        return self.state

    def render(self):
        fig = plt.figure(1)
        ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
        ax.scatter(self.plot_x, self.plot_y, self.plot_z, c='r')  # 绘制数据点,颜色是红色
        ax.scatter(self.t_x, self.t_y, self.t_z, c='b')  # 绘制数据点,颜色是红色
        ax.set_zlabel('Z')  # 坐标轴
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
        plt.draw()
        plt.pause(1)
        plt.close(1)
def stateToQd(x):
    # Converts qd struct used in hardware to x vector used in simulation
    # x is 1 x 13 vector of state variables [pos vel quat omega]
    # qd is a struct including the fields pos, vel, euler, and omega

    # current state
    pos = x[0:3]
    vel = x[3:6]
    Rot = QuatToRot(x[6:10].T)
    [phi, theta, yaw] = RotToRPY_ZXY(Rot)

    euler = [phi, theta, yaw]
    omega = x[10:13]
    return [pos.tolist(), vel.tolist(), euler, omega.tolist()]


def QuatToRot(q):
    # QuatToRot Converts a Quaternion to Rotation matrix
    # normalize q

    q = q / np.sqrt(sum(np.multiply(q, q)))
    qahat = np.zeros([3, 3])
    qahat[0, 1] = -q[3]
    qahat[0, 2] = q[2]
    qahat[1, 2] = -q[1]
    qahat[1, 0] = q[3]
    qahat[2, 0] = -q[2]
    qahat[2, 1] = q[1]
    R = np.eye(3) + 2 * np.dot(qahat, qahat) + 2 * np.dot(q[0], qahat)
    return R


def RPYtoRot_ZXY(phi, theta, psi):
    R = [[math.cos(psi) * math.cos(theta) - math.sin(phi) * math.sin(psi) * math.sin(theta),
          math.cos(theta) * math.sin(psi) + math.cos(psi) * math.sin(phi) * math.sin(theta),
          - math.cos(phi) * math.sin(theta)],
         [- math.cos(phi) * math.sin(psi),
          math.cos(phi) * math.cos(psi),
          math.sin(phi)],
         [math.cos(psi) * math.sin(theta) + math.cos(theta) * math.sin(phi) * math.sin(psi),
          math.sin(psi) * math.sin(theta) - math.cos(psi) * math.cos(theta) * math.sin(phi),
          math.cos(phi) * math.cos(theta)]]
    return R


def RotToRPY_ZXY(R):
    # RotToRPY_ZXY Extract Roll, Pitch, Yaw from a world-to-body Rotation Matrix
    # The rotation matrix in this function is world to body [bRw] you will
    # need to transpose the matrix if you have a body to world [wRb] such
    # that [wP] = [wRb] * [bP], where [bP] is a point in the body frame and
    # [wP] is a point in the world frame
    # bRw = [ cos(psi)*cos(theta) - sin(phi)*sin(psi)*sin(theta),
    #           cos(theta)*sin(psi) + cos(psi)*sin(phi)*sin(theta),
    #          -cos(phi)*sin(theta)]
    #         [-cos(phi)*sin(psi), cos(phi)*cos(psi), sin(phi)]
    #         [ cos(psi)*sin(theta) + cos(theta)*sin(phi)*sin(psi),
    #            sin(psi)*sin(theta) - cos(psi)*cos(theta)*sin(phi),
    #            cos(phi)*cos(theta)]

    phi = math.asin(R[1, 2])
    psi = math.atan2(-R[1, 0] / math.cos(phi), R[1, 1] / math.cos(phi))
    theta = math.atan2(-R[0, 2] / math.cos(phi), R[2, 2] / math.cos(phi))
    return [phi, theta, psi]


def RotToQuat(R):
    # ROTTOQUAT Converts a Rotation matrix into a Quaternion
    # takes in W_R_B rotation matrix
    tr = R[0, 0] + R[1, 1] + R[2, 2]

    if (tr > 0):
        S = np.sqrt(tr + 1.0) * 2  # S=4*qw
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S

    elif ((R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2])):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S=4*qx
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S

    elif (R[1, 1] > R[2, 2]):
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S=4*qy
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S=4*qz
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    q = [qw, qx, qy, qz]
    q = np.multiply(q, np.sign(qw))
    return q


def Tapfly(state):
    r1 = 2 * (0.5 - abs(state[0]-0.5))  # MIN -3 MAX 1
    r2 = 2 * (0.5 - abs(state[1]-0.5))  # MIN -3 MAX 1
    r3 = 2 * (0.5 - abs((state[2] - 1)))  # MIN -2   MAX 1
    r = np.sign(r1) * ((10 * r1) ** 2) + np.sign(r2) * ((10 * r2) ** 2) + np.sign(r3) * ((10 * r3) ** 2)
    return r

def Tapfly(state, target):
    r = (state[0] - target[0]) ** 2 + (state[1] - target[1]) ** 2 + (state[2] - target[2]) ** 2
    survive_bunos = 0.25
    return -r + survive_bunos


def Circle_tracking(state, t):
    target = trajectory('circle', t)
    pos = np.array([state[0], state[1], state[2]])
    vel = np.array([state[3], state[4], state[5]])
    r = -np.linalg.norm(pos - np.array(target[0]))
    # r=1/(np.linalg.norm(pos-np.array(target[0]))+0.1) + 0.1/(np.linalg.norm(vel-np.array(target[1]))+0.1)
    return r + 1


def trajectory(name, t):
    if name == 'circle':
        time_tol = 12
        radius = 5
        dt = 0.0001

        def tj_from_line(start_pos, end_pos, time_ttl, t_c):
            v_max = (end_pos - start_pos) * 2 / time_ttl
            if t_c >= 0 and t_c < time_ttl / 2:
                vel = v_max * t_c / (time_ttl / 2)
                pos = start_pos + t_c * vel / 2
                acc = [0, 0, 0]
            else:
                vel = v_max * (time_ttl - t_c) / (time_ttl / 2)
                pos = end_pos - (time_ttl - t_c) * vel / 2
                acc = [0, 0, 0]
            return [pos, vel, acc[0], acc[1], acc[2]]

        def pos_from_angle(a):
            # pos = [radius*np.cos(a), radius*np.sin(a), 2.5*a/(2*np.pi)]
            pos = [np.multiply(radius, np.cos(a)), np.multiply(radius, np.sin(a)),
                   np.multiply(2.5 / (2 * np.pi), a)]
            return pos

        def get_vel(t):
            angle1 = tj_from_line(0, 2 * np.pi, time_tol, t)[0]
            pos1 = pos_from_angle(angle1)
            angle2 = tj_from_line(0, 2 * np.pi, time_tol, t + dt)[0]
            pos2 = pos_from_angle(angle2)
            vel = (np.subtract(pos2, pos1)) / dt
            return vel

        if t > time_tol:
            pos = [radius, 0, 2.5]
            vel = [0, 0, 0]
            acc = [0, 0, 0]
        else:
            angle = tj_from_line(0, 2 * np.pi, time_tol, t)[0]
            pos = pos_from_angle(angle)
            vel = get_vel(t).tolist()
            acc = ((get_vel(t + dt) - get_vel(t)) / dt).tolist()
        yaw = 0
        yawdot = 0

        desired_state = [pos, vel, acc, yaw, yawdot]

    elif name == 'hover':
        time_tol = 1000
        length = 5
        if t <= 0:
            pos = [0, 0, 0]
            vel = [0, 0, 0]
            acc = [0, 0, 0]
        else:
            pos = [1, 0, 0]
            vel = [0, 0, 0]
            acc = [0, 0, 0]

        yaw = 0
        yawdot = 0

        desired_state = [pos, vel, acc, yaw, yawdot]

    return desired_state


def Circle_Task(state, target_dist):
    altitude_reward = 1 / (abs(state[2] - 1) + 1)
    # if state[2]<=0.25:
    #     contact_loss=10
    # else:
    #     contact_loss=-1
    x, y = state[0], state[1]
    dx, dy = state[3], state[4]
    # print(state)
    # dx, dy = np.sign(state[3]), np.sign(state[4])
    reward = -y * dx + x * dy
    # reward = -y * min(max(dx, -2),2) + x *min(max(dy, -2),2)
    reward /= (0.1 + np.abs(np.sqrt(x ** 2 + y ** 2) - target_dist))
    reward = reward + 0
    return reward


def Hover(state):
    Rot = QuatToRot(state[6:10].T)
    [phi, theta, yaw] = RotToRPY_ZXY(Rot)

    r_phi = (0.3 - abs(phi)) / 0.3
    r_theta = (0.3 - abs(theta)) / 0.3
    r_yaw = (0.01 - abs(yaw)) / 0.01
    d1 = (0.5 - abs(state[0] - 1)) / 0.5  # MIN -3 MAX 1
    d2 = (0.5 - abs(state[1] - 1)) / 0.5  # MIN -3 MAX 1
    d3 = 2 * (0.5 - abs((state[2] - 1))) / 0.5  # MIN -2   MAX 1

    distance = d1 + d2 + d3
    stability = r_phi + r_phi + r_yaw
    # r = np.sign(r1) * ((2*r1) ** 2) + np.sign(r2) * ((2*r2) ** 2) + np.sign(r3) * ((2*r3) ** 2)+r0+1
    # r = min(1.,t*10)*(np.sign(r1) * ((10*r1) ** 2) + np.sign(r2) * ((10*r2) ** 2) + np.sign(r3) * ((10**r3) ** 2))+1/(abs(phi)+0.1) +1/(abs(theta)+0.1)+1/(abs(yaw)+0.01)
    r = np.sign(stability) * ((stability) ** 2) + 0.5 * np.sign(distance) * ((distance) ** 2)
    # r=(1-abs(state[2]-1))**2+(1-abs(state[0]-1))**2+(1-abs(state[1]-1))**2+r1+1
    return r + 1