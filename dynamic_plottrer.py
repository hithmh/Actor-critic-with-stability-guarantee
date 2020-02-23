import pandas as pd
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
ROOT_DIR = './log'


COLORS = ['blue', 'brown', 'red','green','deepskyblue', 'gold','darkorange','cyan', 'magenta','darkred',  'yellow', 'black', 'purple', 'pink',
          'teal',  'lightblue', 'orange', 'lavender', 'turquoise','lime',
        'darkgreen', 'tan',  'gold']

# COLORS_map = {'PPO':'steelblue', 'LPPO':'forestgreen', 'LSAC':'brown', 'SAC':'red','SSAC':'olivedrab','SPPO':'gold',
#           'LAC':'red',
#           }
COLORS_map = {
    #fig1

    #fig2
    'original':'orange',
    'K_i=5':'blue',
    'K_i=10':'deepskyblue',
    'a_i=3.2':'green',
    'a_i=4.8':'brown',
    'noise level=0.5':'red',
    'noise level=1':'gold',
    }

label_fontsize = 10
tick_fontsize = 14
linewidth = 3
markersize = 10





def read_csv(fname):
    return pd.read_csv(fname, index_col=None, comment='#')

def load_results(args,alg_list, contents, env,rootdir=ROOT_DIR):
    # if isinstance(rootdir, str):
    #     rootdirs = [osp.expanduser(rootdir)]
    # else:
    #     dirs = [osp.expanduser(d) for d in rootdir]
    results = {}
    for name in env:
        results[name] = {}
    exp_dirs = os.listdir(rootdir)

    for exp_dir in exp_dirs:

        if exp_dir in env:

            exp_path = os.path.join(rootdir, exp_dir)
            alg_dirs = os.listdir(exp_path)
            for alg_dir in alg_dirs:

                if alg_dir in alg_list:
                    alg_path = os.path.join(exp_path, alg_dir)

                    result = read_eval_data(alg_path, args)

                    results[exp_dir][alg_dir] = result

                else:
                    continue


    return results


def read_eval_data(alg_path, args):

    # under the 'eval' directory

    path = alg_path + '/eval' +'/dynamic'
    evals = os.listdir(path)
    result = {}
    for trial in evals:

        if trial not in args['plot_list']:

            continue

        full_path = os.path.join(path, trial)
        try:
            data = read_csv(full_path + '/progress.csv')
        except pd.errors.EmptyDataError:
            print(alg_path + 'reading csv failed')
            continue
        result[trial]={}
        result[trial]['s']=data['average_path'].values
        result[trial]['s_std']=data['std_path'].values
        result[trial]['r']=data['reference'].values

    return result




def plot_line(results, alg, exp, args):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)


    for i,trial in enumerate(results.keys()):
        # fig = plt.figure(figsize=(9, 6))
        # ax = fig.add_subplot(111)
        if trial in COLORS_map.keys():
            color = COLORS_map[trial]
        else:
            color = COLORS[i]
        t = range(len(results[trial]['s']))
        ax.plot(t, results[trial]['s'], color=color, label=trial)
        ax.fill_between(t, results[trial]['s'] - results[trial]['s_std'], results[trial]['s'] + results[trial]['s_std'],
                        color=color, alpha=.3)
        ax.plot(t, results[trial]['r'], color=color,linestyle='dashed')
        if args['labels_on']:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, fontsize=20, loc=2, fancybox=False, shadow=False)
        # plt.savefig('-'.join([exp, alg, trial, 'dynamic.pdf']))
        # plt.show()
    if args['labels_on']:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize=20, loc=2, fancybox=False, shadow=False)
    plt.savefig('-'.join([exp , alg,'dynamic.pdf']))
    plt.show()

    return

def main(args, alg_list, measure_list, env):
    results = load_results(args, alg_list, measure_list, env)

    for exp in results.keys():
        for alg in alg_list:
            plot_line(results[exp][alg], alg, exp, args)

    return

if __name__ == '__main__':

    alg_list = [
        # 'LAC',
        # 'LAC-horizon=5-quadratic',
        # 'SAC',
        'SPPO',
        ]

    args = {
        'data': ['training', 'eval'][1],
        'eval_content': [
            'K=1'
        ],
        'labels_on':True,
        # 'labels_on': False,
        'plot_list': [
            # '1',
            # '2',
            # '3',
            # '4',
            # '5',
            # 'original',
            # 'K_i=5',
            # 'K_i=10',
            # 'a_i=3.2',
            # 'a_i=4.8',
            # 'delta=0.5',
            # 'delta=1',
            # 'noise level=1',
            # 'noise level=0.5',
            # 'K_i=5',
            # 'K_i=10',
            # 'a_i=3.2',
            # 'a_i=4.8',
            'original',
            'length=1.0',
            'length=1.5',
            'length=2.0',
        ],
        'formal_plot':True,
        # 'formal_plot': False,
        # 'ylim':[0,200],
        }

    content = [
        'return',
        # 'eprewmean',
        # 'death_rate',
        # 'return_std',
        # 'average_length'
    ]
    env = [
        # 'oscillator',
        # 'MJS2',
        # 'MJS1',
        'cartpole_cost'
    ]
    main(args, alg_list, content, env)

