import gym
import datetime
import numpy as np
import ENV.env
SEED = None

VARIANT = {
    # 'env_name': 'FetchReach-v1',
    # 'env_name': 'Antcost-v0',
    # 'env_name': 'oscillator',
    # 'env_name': 'MJS1',
    'env_name': 'minitaur',
    # 'env_name': 'swimmer',
    # 'env_name': 'racecar',
    # 'env_name': 'MJS2',
    # 'env_name': 'oscillator_complicated',
    # 'env_name': 'HalfCheetahcost-v0',
    # 'env_name': 'cartpole_cost',
    #training prams
    'algorithm_name': 'LAC',
    # 'algorithm_name': 'SAC_cost',
    # 'algorithm_name': 'SPPO',
    # 'algorithm_name': 'DDPG',
    # 'algorithm_name': 'CPO',

    # 'additional_description': '-N=50',
    # 'additional_description': '-64-64',
    # 'additional_description': '-horizon=5-alpha3=.1',
    'additional_description': '-alpha=.1',
    # 'additional_description': '-pos-track-alpha=1.',
    # 'additional_description': '-pos-track-low-lambda',
    # 'additional_description': '-trial',
    # 'evaluate': False,
    'train': True,
    # 'train': False,

    'num_of_trials': 10,   # number of random seeds
    'num_of_evaluation_paths': 10,  # number of rollouts for evaluation
    'num_of_training_paths': 10,  # number of training rollouts stored for analysis
    'start_of_trial': 0,

    #evaluation params
    'evaluation_form': 'constant_impulse',
    # 'evaluation_form': 'dynamic',
    # 'evaluation_form': 'impulse',
    # 'evaluation_form': 'various_disturbance',
    # 'evaluation_form': 'param_variation',
    # 'evaluation_form': 'trained_disturber',
    'eval_list': [
        # cartpole
        # 'LAC-horizon=3-alpha3=.1',
        # 'LAC-horizon=inf-alpha3=.1',
        # 'SAC_cost-64-64',
        # 'SAC-video',
        # 'SAC',
        # 'LAC-horizon=5-quadratic',
        # 'LQR',
        # 'SAC_cost-new',
        # halfcheetah
        # 'LAC-des=1-horizon=inf-alpha=1',
        # 'LAC-des=1-horizon=inf',
        # 'LAC',
        # 'SAC_cost',

        # ant
        # 'LAC-des=1-horizon=inf-alpha=1',
        # 'SAC_cost-des=1-no_contrl_cost',

        # Fetch
        # 'SPPO',
        # 'LAC',
        # 'LAC-relu',
        # 'LAC-biquad',
        # 'SAC',
        # 'LAC-pos-track',
        # 'SAC_cost-pos-track',
        # 'SAC_cost-0.75-new',
        #oscillator
        # 'LAC',
        # 'SAC_cost',
        'SPPO',
    ],
    'trials_for_eval': [str(i) for i in range(0, 3)],

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
        'eval_render': False,
        'network_structure':
            {'critic': [64, 64, 16],
             'actor': [64,64],
             },
    },
    'swimmer': {
        'max_ep_steps': 250,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'disturbance dim': 1,
        'eval_render': False,
        'network_structure':
            {'critic': [64, 64, 16],
             'actor': [64, 64],
             },
    },
    'oscillator': {
        'max_ep_steps': 400,
        'max_global_steps': int(1e5),
        'max_episodes': int(1e5),
        'disturbance dim': 2,
        'eval_render': False,
        'network_structure':
            {'critic': [256, 256, 16],
             'actor': [64, 64],
             },
    },
    'MJS1': {
        'max_ep_steps': 400,
        'max_global_steps': int(2e5),
        'max_episodes': int(2e5),
        'disturbance dim': 1,
        'eval_render': False,
        'network_structure':
            {'critic': [64, 64, 16],
             'actor': [64,64],
             },
    },
    'MJS2': {
        'max_ep_steps': 400,
        'max_global_steps': int(2e5),
        'max_episodes': int(2e5),
        'disturbance dim': 1,
        'eval_render': False,
        'network_structure':
            {'critic': [64, 64, 16],
             'actor': [64,64],
             },
    },
    'racecar': {
        'max_ep_steps': 20,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'disturbance dim': 1,
        'eval_render': True,
        'network_structure':
            {'critic': [64, 64, 16],
             'actor': [64, 64],
             },
    },
    'minitaur': {
        'max_ep_steps': 500,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'disturbance dim': 1,
        'eval_render': False,
        # 'network_structure':
        #     {'critic': [98, 85, 16],
        #      'actor': [185,95],
        #      },
        'network_structure':
            {'critic': [256, 256, 16],
             'actor': [64,64],
             },
    },
    'oscillator_complicated': {
        'max_ep_steps': 400,
        'max_global_steps': int(1e5),
        'max_episodes': int(2e5),
        'disturbance dim': 2,
        'eval_render': False,
        'network_structure':
            {'critic': [64, 64, 16],
             'actor': [64, 64],
             },
    },
    'HalfCheetahcost-v0': {
        'max_ep_steps': 200,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'disturbance dim': 6,
        'eval_render': False,
        'network_structure':
            {'critic': [256, 256, 16],
             'actor': [64, 64],
             },
    },
    'Quadrotorcost-v0': {
        'max_ep_steps': 2000,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'eval_render': False,
        'network_structure':
            {'critic': [64, 64, 16],
             'actor': [64, 64],
             },
    },
    'Antcost-v0': {
        'max_ep_steps': 200,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'disturbance dim': 8,
        'eval_render': False,
        'network_structure':
            {'critic': [64, 64, 16],
             'actor': [64, 64],
             },
    },
    'FetchReach-v1': {
        # 'max_ep_steps': 50,
        'max_ep_steps': 200,
        'max_global_steps': int(3e5),
        'max_episodes': int(1e6),
        'disturbance dim': 4,
        'eval_render': False,
        'network_structure':
            {'critic': [64, 64, 16],
             'actor': [64, 64],
             },
    },
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
        'alpha3': .1,
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
        'value_horizon': 2,
        # 'finite_horizon': True,
        'finite_horizon': False,
        'soft_predict_horizon': False,
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
    # 'SPPO': {
    #     'batch_size':10000,
    #     'output_format':['csv'],
    #     'gae_lamda':0.95,
    #     'safety_gae_lamda':0.5,
    #     'labda': 1.,
    #     'number_of_trajectory':10,
    #     'alpha3': 0.1,
    #     'lr_c': 1e-3,
    #     'lr_a': 1e-4,
    #     'gamma': 0.995,
    #     'cliprange':0.2,
    #     'delta':0.01,
    #     'd_0': 1,
    #     'form_of_lyapunov': 'l_reward',
    #     'safety_threshold': 0.,
    #     'use_lyapunov': False,
    #     'use_adaptive_alpha3': False,
    #     'use_baseline':False,
    #     },
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
        'num_of_paths': 100,   # number of path for evaluation
        'impulse_instant': 200,
    },
    'constant_impulse': {
        # 'magnitude_range': np.arange(120, 125, 5),
        # 'magnitude_range': np.arange(80, 155, 5),
        # 'magnitude_range': np.arange(80, 155, 5),
        # 'magnitude_range': np.arange(80, 155, 5),
        # 'magnitude_range': np.arange(0.2, 2.2, .2),
        'magnitude_range': np.arange(0.1, 1.0, .1),
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
        'additional_description': 'original',
        'num_of_paths': 20,   # number of path for evaluation
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
    elif name == 'cartpole_cost_v2':
        from envs.ENV_V2 import CartPoleEnv_adv as dreamer
        env = dreamer()
        env = env.unwrapped
    elif name == 'cartpole_cost_partial':
        from envs.ENV_V3 import CartPoleEnv_adv as dreamer
        env = dreamer()
        env = env.unwrapped
    elif name == 'cartpole_cost_real':
        from envs.ENV_V4 import CartPoleEnv_adv as dreamer
        env = dreamer()
        env = env.unwrapped
    elif name == 'cartpole_cost_swing_up':
        from envs.ENV_V5 import CartPoleEnv_adv as dreamer
        env = dreamer()
        env = env.unwrapped
    elif name == 'cartpole_cost_real_no_friction':
        from envs.ENV_V6 import CartPoleEnv_adv as dreamer
        env = dreamer()
        env = env.unwrapped
    elif name == 'cartpole_cost_with_motor':
        from envs.ENV_V7 import CartPoleEnv_adv as dreamer
        env = dreamer()
        env = env.unwrapped
    elif name == 'cartpole_cost_with_fitted_motor':
        from envs.ENV_V8 import CartPoleEnv_adv as dreamer
        env = dreamer(eval=True)
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
    elif name == 'Quadrotorcost-v0':
        env = gym.make('Quadrotorcons-v0')
        env = env.unwrapped
        env.modify_action_scale = False
        env.use_cost = True
    elif name == 'minitaur':
        from envs.minitaur_env import minitaur_env as env
        env = env(render=VARIANT['env_params']['eval_render'])
        env = env.unwrapped
    elif name == 'racecar':
        from envs.racar_env import racecar_env as env
        env = env(renders=VARIANT['env_params']['eval_render'])
        env = env.unwrapped
    elif name == 'swimmer':
        from envs.swimmer import swimmer_env as env
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


