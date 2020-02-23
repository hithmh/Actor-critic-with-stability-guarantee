import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import copy
def get_evaluation_rollouts(policy, env, num_of_paths, max_ep_steps, render= True):

    a_bound = env.action_space.high
    paths = []

    for ep in range(num_of_paths):
        s = env.reset()
        path = {'rewards':[],
                'lrewards':[]}
        for step in range(max_ep_steps):
            if render:
                env.render()
            a = policy.choose_action(s, evaluation=True)
            action = a * a_bound
            action = np.clip(action, -a_bound, a_bound)
            s_, r, done, info = env.step(action)
            l_r = info['l_rewards']

            path['rewards'].append(r)
            path['lrewards'].append(l_r)
            s = s_
            if done or step == max_ep_steps-1:
                paths.append(path)
                break
    if len(paths)< num_of_paths:
        print('no paths is acquired')

    return paths


def evaluate_rollouts(paths):
    total_returns = [np.sum(path['rewards']) for path in paths]
    total_lreturns = [np.sum(path['lrewards']) for path in paths]
    episode_lengths = [len(p['rewards']) for p in paths]
    import matplotlib.pyplot as plt
    [plt.plot(np.arange(0, len(path['rewards'])), path['rewards']) for path in paths]
    try:
        diagnostics = OrderedDict((
            ('return-average', np.mean(total_returns)),
            ('return-min', np.min(total_returns)),
            ('return-max', np.max(total_returns)),
            ('return-std', np.std(total_returns)),
            ('lreturn-average', np.mean(total_lreturns)),
            ('lreturn-min', np.min(total_lreturns)),
            ('lreturn-max', np.max(total_lreturns)),
            ('lreturn-std', np.std(total_lreturns)),
            ('episode-length-avg', np.mean(episode_lengths)),
            ('episode-length-min', np.min(episode_lengths)),
            ('episode-length-max', np.max(episode_lengths)),
            ('episode-length-std', np.std(episode_lengths)),
        ))
    except ValueError:
        print('Value error')
    else:
        return diagnostics


def evaluate_training_rollouts(paths):
    data = copy.deepcopy(paths)
    if len(data) < 1:
        return None
    try:
        diagnostics = OrderedDict((
            ('return', np.mean([np.sum(path['rewards']) for path in data])),
            ('length', np.mean([len(p['rewards']) for p in data])),
        ))
    except KeyError:
        return
    [path.pop('rewards') for path in data]
    for key in data[0].keys():
        result = [np.mean(path[key]) for path in data]
        diagnostics.update({key: np.mean(result)})

    return diagnostics
