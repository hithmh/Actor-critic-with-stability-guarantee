
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time
from pool.pool import Pool
from .utils import evaluate_training_rollouts
from collections import OrderedDict, deque
import logger
from variant import *
from robustness_eval import training_evaluation


###############################  DDPG  ####################################
class SDDPG(object):
    def __init__(self,
                 a_dim,
                 s_dim,
                 variant,
                 ):



        ###############################  Model parameters  ####################################
        self.memory_capacity = variant['memory_capacity']
        self.cons_memory_capacity = variant['cons_memory_capacity']
        self.batch_size = variant['batch_size']
        gamma = variant['gamma']
        tau = variant['tau']
        self.pointer = 0
        self.cons_pointer = 0
        self.sess = tf.Session()
        self.a_dim, self.s_dim, = a_dim, s_dim,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.a_input = tf.placeholder(tf.float32, [None, a_dim], 'a_input')
        self.LR_A = tf.placeholder(tf.float32, None, 'LR_A')
        self.LR_C = tf.placeholder(tf.float32, None, 'LR_C')
        self.terminal = tf.placeholder(tf.float32, [None, 1], 'terminal')
        self.a = self._build_a(self.S, )  # 这个网络用于及时更新参数
        self.q = self._build_c(self.S, self.a)  # 这个网络是用于及时更新参数
        self.noise_scale = tf.placeholder(tf.float32, None, 'noise_scale')

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')

        ###############################  Model Learning Setting  ####################################
        ema = tf.train.ExponentialMovingAverage(decay=1 - tau)  # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]  # soft update operation

        # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)  # replaced target parameters

        # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
        q_ = self._build_c(self.S_, tf.stop_gradient(a_), reuse=True, custom_getter=ema_getter)



        a_loss = tf.reduce_mean(self.q)

        self.atrain = tf.train.AdamOptimizer(self.LR_A).minimize(a_loss,
                                                                 var_list=a_params)  # 以learning_rate去训练，方向是minimize loss，调整列表参数，用adam
        self.sigma = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(self.a_dim), scale_diag=tf.ones(self.a_dim))
        self.sample_action_op = self.a + self.noise_scale * self.sigma.sample(tf.shape(self.S)[0])
        with tf.control_dependencies(target_update):  # soft replacement happened at here
            q_target = self.R + gamma*(1-self.terminal) * tf.stop_gradient(q_)   # ddpg

            self.td_error = tf.losses.mean_squared_error(labels=q_target, predictions=self.q)

            self.ctrain = tf.train.AdamOptimizer(self.LR_C).minimize(self.td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.opt = [self.atrain, self.ctrain, ]
        self.diagnotics = [self.td_error, a_loss]



    def choose_action(self, s, evaluation=False, noise_scale=0):
        a = self.sess.run(self.sample_action_op, {self.S: s[np.newaxis, :], self.noise_scale: noise_scale})[0]
        if evaluation:
            a = np.tanh(a)
        return a


    def learn(self, LR_A, LR_C, batch):
        bs = batch['s']  # state
        ba = batch['a']  # action

        br = batch['r']  # reward
        bterminal = batch['terminal']
        bs_ = batch['s_']  # next state
        feed_dict = {self.a_input: ba, self.S: bs, self.S_: bs_, self.R: br, self.terminal: bterminal,
                     self.LR_C: LR_C, self.LR_A: LR_A}

        self.sess.run(self.opt, feed_dict)
        td_error, a_loss= self.sess.run(self.diagnotics, feed_dict)

        return td_error, a_loss


    #action 选择模块也是actor模块


    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net_0 = tf.layers.dense(s, 256, activation=tf.nn.relu, name='l1', trainable=trainable)#原始是30
            net_1 = tf.layers.dense(net_0, 128, activation=tf.nn.relu, name='l2', trainable=trainable)  # 原始是30
            a = tf.layers.dense(net_1, self.a_dim, name='a', trainable=trainable)
            return a

    # def _build_a(self, s, reuse=None, custom_getter=None):
    #     trainable = True
    #     with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
    #         net_0 = tf.layers.dense(s, 256, activation=tf.nn.relu, name='l1', trainable=trainable)#原始是30
    #         net_1 = tf.layers.dense(net_0, 128, activation=tf.nn.relu, name='l2', trainable=trainable)  # 原始是30
    #         a = tf.layers.dense(net_1, self.a_dim, activation=None, name='a', trainable=trainable)
    #         return a


    #critic模块
    def _build_c(self, s, a, name ='Critic', reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope(name, reuse=reuse, custom_getter=custom_getter):
            n_l1 = 256#30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net_0 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net_1 = tf.layers.dense(net_0, 256, activation=tf.nn.relu, name='l2', trainable=trainable)  # 原始是30
            return tf.layers.dense(net_1, 1, trainable=trainable)  # Q(s,a)

    def _build_l(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Lyapunov', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 256#30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net_0 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net_1 = tf.layers.dense(net_0, 256, activation=tf.nn.relu, name='l2', trainable=trainable)  # 原始是30
            return tf.layers.dense(net_1, 1, trainable=trainable)  # Q(s,a)


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

    env_params = variant['env_params']

    max_episodes = env_params['max_episodes']
    max_ep_steps = env_params['max_ep_steps']
    max_global_steps = env_params['max_global_steps']
    store_last_n_paths = variant['store_last_n_paths']
    evaluation_frequency = variant['evaluation_frequency']

    policy_params = variant['alg_params']

    min_memory_size = policy_params['min_memory_size']
    steps_per_cycle = policy_params['steps_per_cycle']
    train_per_cycle = policy_params['train_per_cycle']
    noise_scale = policy_params['noise']
    noise_scale_now = noise_scale
    batch_size = policy_params['batch_size']

    lr_a, lr_c = policy_params['lr_a'], policy_params['lr_c']
    lr_a_now = lr_a  # learning rate for actor
    lr_c_now = lr_c  # learning rate for critic


    if 'Fetch' in env_name or 'Hand' in env_name:
        s_dim = env.observation_space.spaces['observation'].shape[0]\
                + env.observation_space.spaces['achieved_goal'].shape[0]+ \
                env.observation_space.spaces['desired_goal'].shape[0]
    else:
        s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    # if disturber_params['process_noise']:
    #     d_dim = disturber_params['noise_dim']
    # else:
    #     d_dim = env_params['disturbance dim']

    a_upperbound = env.action_space.high
    a_lowerbound = env.action_space.low
    policy = SDDPG(a_dim,s_dim, policy_params)

    pool_params = {
        's_dim': s_dim,
        'a_dim': a_dim,
        'd_dim': 1,
        'store_last_n_paths': store_last_n_paths,
        'memory_capacity': policy_params['memory_capacity'],
        'min_memory_size': policy_params['min_memory_size'],
        # 'history_horizon': policy_params['history_horizon'],
        # 'finite_horizon':policy_params['finite_horizon']
    }
    if 'value_horizon' in policy_params.keys():
        pool_params.update({'value_horizon': policy_params['value_horizon']})
    else:
        pool_params['value_horizon'] = None
    pool = Pool(pool_params)
    # For analyse
    Render = env_params['eval_render']

    # Training setting
    t1 = time.time()
    global_step = 0
    last_training_paths = deque(maxlen=store_last_n_paths)
    training_started = False

    log_path = variant['log_path']
    logger.configure(dir=log_path, format_strs=['csv'])
    logger.logkv('tau', policy_params['tau'])

    logger.logkv('batch_size', policy_params['batch_size'])


    for i in range(max_episodes):

        current_path = {'rewards': [],
                        'a_loss': [],
                        }

        if global_step > max_global_steps:
            break

        s = env.reset()
        if 'Fetch' in env_name or 'Hand' in env_name:
            s = np.concatenate([s[key] for key in s.keys()])

        for j in range(max_ep_steps):
            if Render:
                env.render()
            a = policy.choose_action(s, noise_scale=noise_scale_now)
            action = np.tanh(a)
            action = a_lowerbound + (action + 1.) * (a_upperbound - a_lowerbound) / 2


            s_, r, done, info = env.step(action)

            if 'Fetch' in env_name or 'Hand' in env_name:
                s_ = np.concatenate([s_[key] for key in s_.keys()])
                if info['done'] > 0:
                    done = True

            if training_started:
                global_step += 1

            if j == max_ep_steps - 1:
                done = True

            terminal = 1. if done else 0.
            pool.store(s, a, np.zeros([1]), np.zeros([1]), r, terminal, s_)
            # policy.store_transition(s, a, disturbance, r,0, terminal, s_)

            if pool.memory_pointer > min_memory_size and global_step % steps_per_cycle == 0:
                training_started = True

                for _ in range(train_per_cycle):
                    batch = pool.sample(batch_size)
                    td_error, a_loss = policy.learn(lr_a_now, lr_c_now, batch)

            if training_started:
                current_path['rewards'].append(r)
                current_path['a_loss'].append(a_loss)



            if training_started and global_step % evaluation_frequency == 0 and global_step > 0:

                logger.logkv("total_timesteps", global_step)

                training_diagnotic = evaluate_training_rollouts(last_training_paths)
                if training_diagnotic is not None:
                    eval_diagnotic = training_evaluation(variant, env, policy)
                    [logger.logkv(key, eval_diagnotic[key]) for key in eval_diagnotic.keys()]
                    training_diagnotic.pop('return')
                    [logger.logkv(key, training_diagnotic[key]) for key in training_diagnotic.keys()]
                    logger.logkv('lr_a', lr_a_now)
                    logger.logkv('lr_c', lr_c_now)

                    string_to_print = ['time_step:', str(global_step), '|']
                    [string_to_print.extend([key, ':', str(eval_diagnotic[key]), '|'])
                     for key in eval_diagnotic.keys()]
                    [string_to_print.extend([key, ':', str(round(training_diagnotic[key], 2)) , '|'])
                     for key in training_diagnotic.keys()]
                    print(''.join(string_to_print))

                logger.dumpkvs()
            # 状态更新
            s = s_

            # OUTPUT TRAINING INFORMATION AND LEARNING RATE DECAY
            if done:
                if training_started:
                    last_training_paths.appendleft(current_path)

                frac = 1.0 - (global_step - 1.0) / max_global_steps
                noise_scale_now = noise_scale * frac
                lr_a_now = lr_a * frac  # learning rate for actor
                lr_c_now = lr_c * frac  # learning rate for critic

                break
    policy.save_result(log_path)

    print('Running time: ', time.time() - t1)
    return