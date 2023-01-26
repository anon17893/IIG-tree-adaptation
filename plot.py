import yaml
import argparse
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from cycler import cycler
import os

from agents.utils import ExperimentGenerator


GAME_SCALES = {'kuhn_poker': 4.0, 'leduc_poker': 26.0, 'liars_dice': 2.0}

def plot_ax(ax, df, y_name, y_label = ''):
    sns.lineplot(data=df, x='step', y=y_name, hue='Algorithm', style='Algorithm', ax=ax)   
    ax.grid()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Episode')
    ax.set_ylabel(y_label)

def make_df(y_name, game_name, results, exp_generator):
    columns = ['step',y_name,'Algorithm']
    df = pd.DataFrame(columns=columns)
    for agent_name in exp_generator.agent_names:
        step = results[game_name][agent_name]['step']
        alg = [agent_name]*len(step)
        for  y in results[game_name][agent_name][y_name]:
            _df = pd.DataFrame(data = dict(zip(columns, [step, y / GAME_SCALES[game_name], alg])))
            df = pd.concat([df,_df], ignore_index=True)
    return df


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, 
                        help='Path to configuration file.')
    args = parser.parse_args()

    config_path = args.config
    exp_config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)


    exp_generator = ExperimentGenerator(**exp_config)
    results = exp_generator.load_results()

    # figure size in inches
    plt.rcParams['figure.figsize'] = 20,7
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams['font.size'] = 14
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['axes.prop_cycle']=(cycler('color', ['r', 'g', 'b', 'y', 'c']) +
                            cycler('linestyle', ['-', '--', ':', '-.', '-']))

    #Plot
    fig, axs  = plt.subplots(nrows=1, ncols=len(exp_generator.game_names))
    for i, game_name in enumerate(exp_generator.game_names):
        y_label = 'Exploitability' if i==0 else ''
        # df = make_df('current', game_name, results, exp_generator)
        # plot_ax(axs[0,i], df, 'current', y_label)
        # axs[0,i].set_title(f'Current exp. in {game_name}'.replace('_',' ').replace('kuhn','Kuhn').replace('leduc','Leduc'))
        df = make_df('average', game_name, results, exp_generator)
        plot_ax(axs[i], df, 'average', y_label)
        axs[i].set_title(f'Average exp. in {game_name}'.replace('_',' ').replace('kuhn','Kuhn').replace('leduc','Leduc'))

    save_path = os.path.join(exp_generator.save_path, exp_generator.description+'_plot.pdf')
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f'Saved plot at {save_path}')
    #fig.show()