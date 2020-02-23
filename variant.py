import gym
import datetime
import numpy as np
import ENV.env
SEED = None

VARIANT = {
    # 'env_name': 'FetchReach-v1',
    # 'env_name': 'oscillator',
    # 'env_name': 'oscillator_complicated',
    # 'env_name': 'HalfCheetahcost-v0',
    'env_name': 'cartpole_cost',
    #training prams
    'algorithm_name': 'LAC',
    # 'algorithm_name': 'SPPO',
    # 'algorithm_name': 'DDPG',
    # 'algorithm_name': 'CPO',
    # 'algorithm_name': 'SAC_cost',
    'additional_description': '',
    # 'evaluate': False,
    'train': True,
    # 'train': False,

    'num_of_trials': 10,   # number of random seeds
    'store_last_n_paths': 10,  # number of trajectories for evaluation during training
    'start_of_trial': 0,

    #evaluation params
    # 'evaluation_form': 'constant_impulse',
    'evaluation_form': 'dynamic',
    # 'evaluation_form': 'impulse',
    # 'evaluation_form': 'various_disturbance',
    # 'evaluation_form': 'param_variation',
    # 'evaluation_form': 'trained_disturber',
    'eval_list': [
    ],
    'trials_for_eval': [str(i) for i in range(0, 10)],

    'evaluation_frequency': 2048,
}
if VARIANT['algorithm_name'] == 'RARL':
    ITA = 0
VARIANT['log_path']='/'.join(['./log', VARIANT['env_name'], VARIANT['algorithm_name'] + VARIANT['additional_description']])

ENV_PARAMS = {
    'cartpole_cost': {
        'max_ep_steps': 250,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'disturbance dim': 1,
        'eval_render': True,},
    'oscillator': {
        'max_ep_steps': 400,
        'max_global_steps': int(1e5),
        'max_episodes': int(1e5),
        'disturbance dim': 2,
        'eval_render': False,},
    'oscillator_complicated': {
        'max_ep_steps': 400,
        'max_global_steps': int(1e5),
        'max_episodes': int(2e5),
        'disturbance dim': 2,
        'eval_render': False,},
    'HalfCheetahcost-v0': {
        'max_ep_steps': 200,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'disturbance dim': 6,
        'eval_render': False,},
    'Quadrotor': {
        'max_ep_steps': 2000,
        'max_global_steps': int(10e6),
        'max_episodes': int(1e6),
        'eval_render': False,},
    'Antcost-v0': {
        'max_ep_steps': 200,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'disturbance dim': 8,
        'eval_render': False,},
    'FetchReach-v1': {
        # 'max_ep_steps': 50,
        'max_ep_steps': 200,
        'max_global_steps': int(3e5),
        'max_episodes': int(1e6),
        'disturbance dim': 4,
        'eval_render': True, },
}
ALG_PARAMS = {
    'MPC':{
        'horizon': 5,
    },

    'LQR':{
        'use_Kalman': False,
    },

    'LAC': {
        'iter_of_actor_train_per_epoch': 50,
        'iter_of_disturber_train_per_epoch': 50,
        'memory_capacity': int(1e6),
        'min_memory_size': 1000,
        'batch_size': 256,
        'labda': 1.,
        'alpha': 2.,
        'alpha3': 1.,
        'tau': 5e-3,
        'lr_a': 1e-4,
        'lr_c': 3e-4,
        'lr_l': 3e-4,
        'gamma': 0.995,
        # 'gamma': 0.75,
        'steps_per_cycle': 100,
        'train_per_cycle': 80,
        'use_lyapunov': True,
        'adaptive_alpha': True,
        'approx_value': True,
        'value_horizon': 5,
        # 'finite_horizon': True,
        'form_of_lyapunov': 'inf',
        # 'form_of_lyapunov': 'finite',
        # 'form_of_lyapunov': 'soft_horizon',
        # 'form_of_lyapunov': 'cost',
        # 'form_of_lyapunov': 'entire_horizon',

        'target_entropy': None,
        'history_horizon': 0,  # 0 is using current state only
    },

    'DDPG': {
        'memory_capacity': int(1e6),
        'cons_memory_capacity': int(1e6),
        'min_memory_size': 1000,
        'batch_size': 256,
        'labda': 1.,
        'alpha3': 0.001,
        'tau': 5e-3,
        'noise': 1.,
        'lr_a': 3e-4,
        'lr_c': 3e-4,
        'gamma': 0.99,
        'steps_per_cycle': 100,
        'train_per_cycle': 80,
        'history_horizon': 0,  # 0 is using current state only
        },
    'SAC_cost': {
        'iter_of_actor_train_per_epoch': 50,
        'iter_of_disturber_train_per_epoch': 50,
        'memory_capacity': int(1e6),
        'cons_memory_capacity': int(1e6),
        'min_memory_size': 1000,
        'batch_size': 256,
        'labda': 1.,
        'alpha': 1.,
        'alpha3': 0.5,
        'tau': 5e-3,
        'lr_a': 1e-4,
        'lr_c': 3e-4,
        'lr_l': 3e-4,
        'gamma': 0.995,
        # 'gamma': 0.75,
        'steps_per_cycle': 100,
        'train_per_cycle': 50,
        'use_lyapunov': False,
        'adaptive_alpha': True,
        'target_entropy': None,

    },

    'SPPO': {
        'batch_size':2000,
        'output_format':['csv'],
        'gae_lamda':0.95,
        'safety_gae_lamda':0.95,
        'labda': 1.,
        'number_of_trajectory':50,
        'alpha3': 0.1,
        'lr_c': 3e-4,
        'lr_a': 1e-4,
        'lr_l': 1e-4,
        'gamma': 0.995,
        'cliprange':0.2,
        'delta':0.01,
        # 'd_0': 1,
        'finite_horizon':False,
        'horizon': 5,
        'form_of_lyapunov': 'l_reward',
        'safety_threshold': 10.,
        'use_lyapunov': False,
        'use_adaptive_alpha3': False,
        'use_baseline':False,
        },
}


EVAL_PARAMS = {
    'param_variation': {
        'param_variables': {
            'mass_of_pole': np.arange(0.05, 0.55, 0.05),  # 0.1
            'length_of_pole': np.arange(0.1, 2.1, 0.1),  # 0.5
            'mass_of_cart': np.arange(0.1, 2.1, 0.1),    # 1.0
            # 'gravity': np.arange(9, 10.1, 0.1),  # 0.1

        },
        'grid_eval': True,
        # 'grid_eval': False,
        'grid_eval_param': ['length_of_pole', 'mass_of_cart'],
        'num_of_paths': 100,   # number of path for evaluation
    },
    'impulse': {
        # 'magnitude_range': np.arange(150, 160, 5),
        'magnitude_range': np.arange(80, 155, 5),
        # 'magnitude_range': np.arange(80, 155, 10),
        # 'magnitude_range': np.arange(0.1, 1.1, .1),
        'num_of_paths': 20,   # number of path for evaluation
        'impulse_instant': 200,
    },
    'constant_impulse': {
        # 'magnitude_range': np.arange(120, 125, 5),
        'magnitude_range': np.arange(80, 155, 5),
        # 'magnitude_range': np.arange(80, 155, 5),
        # 'magnitude_range': np.arange(80, 155, 5),
        # 'magnitude_range': np.arange(0.2, 2.2, .2),
        # 'magnitude_range': np.arange(0.1, 1.0, .1),
        'num_of_paths': 20,   # number of path for evaluation
        'impulse_instant': 20,
    },
    'various_disturbance': {
        'form': ['sin', 'tri_wave'][0],
        'period_list': np.arange(2, 11, 1),
        # 'magnitude': np.array([1, 1, 1, 1, 1, 1]),
        'magnitude': np.array([80]),
        # 'grid_eval': False,
        'num_of_paths': 100,   # number of path for evaluation
    },
    'trained_disturber': {
        # 'magnitude_range': np.arange(80, 125, 5),
        # 'path': './log/cartpole_cost/RLAC-full-noise-v2/0/',
        'path': './log/HalfCheetahcost-v0/RLAC-horizon=inf-dis=.1/0/',
        'num_of_paths': 100,   # number of path for evaluation
    },
    'dynamic': {
        'additional_description': 'video',
        'num_of_paths': 100,   # number of path for evaluation
        # 'plot_average': True,
        'plot_average': False,
        'directly_show': True,
    },
}
VARIANT['env_params']=ENV_PARAMS[VARIANT['env_name']]
VARIANT['eval_params']=EVAL_PARAMS[VARIANT['evaluation_form']]
VARIANT['alg_params']=ALG_PARAMS[VARIANT['algorithm_name']]

RENDER = True
def get_env_from_name(name):
    if name == 'cartpole_cost':
        from envs.ENV_V1 import CartPoleEnv_adv as dreamer
        env = dreamer()
        env = env.unwrapped

    elif name == 'oscillator':
        from envs.oscillator import oscillator as env
        env = env()
        env = env.unwrapped
    elif name == 'MJS1':
        from envs.MJS1 import MJS as env
        env = env()
        env = env.unwrapped
    elif name == 'MJS2':
        from envs.MJS2 import MJS as env
        env = env()
        env = env.unwrapped
    elif name == 'oscillator_complicated':
        from envs.oscillator_complicated import oscillator as env
        env = env()
        env = env.unwrapped
    elif name == 'Quadrotor':
        from envs.quadrotor import QuadEnv2 as env
        env = env()
        env = env.unwrapped

    else:
        env = gym.make(name)
        env = env.unwrapped
        if name == 'Quadrotorcons-v0':
            if 'CPO' not in VARIANT['algorithm_name']:
                env.modify_action_scale = False
        if 'Fetch' in name or 'Hand' in name:
            env.unwrapped.reward_type = 'dense'
    env.seed(SEED)
    return env

def get_train(name):
    if 'RARL' in name:
        from LAC.RARL import train as train
    elif 'LAC' in name:
        from LAC.LAC_V1 import train
    elif 'SPPO' in name:
        from CPO.CPO2 import train
    elif 'DDPG' in name:
        from LAC.SDDPG_V8 import train
    # elif 'CPO' in name:
    #     from CPO.CPO2 import train
    else:
        from LAC.SAC_cost import train

    return train

def get_policy(name):
    if 'RARL' in name:
        from LAC.RARL import RARL as build_func
    elif 'LAC' in name :
        from LAC.LAC_V1 import LAC as build_func
    elif 'LQR' in name:
        from LAC.lqr import LQR as build_func
    elif 'MPC' in name:
        from LAC.MPC import MPC as build_func
    elif 'SPPO' in name:
        from CPO.CPO2 import CPO as build_func
    # elif 'CPO' in name:
    #     from CPO.CPO2 import CPO as build_func
    elif 'DDPG' in name:
        from LAC.SDDPG_V8 import SDDPG as build_func
    else:
        from LAC.SAC_cost import SAC_cost as build_func
    return build_func

def get_eval(name):
    if 'LAC' in name or 'SAC_cost' in name:
        from LAC.LAC_V1 import eval

    return eval


