from __future__ import division
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import glob
from itertools import cycle

from models import *

FIGURES_DIR = 'figures/'
if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)


###########
## UTILS ##
###########
def save_figure(filename, legend=None):
    filename = filename.replace('.', '-')
    filename += '.pdf'
    if legend:
        plt.savefig(
            FIGURES_DIR + filename,
            bbox_extra_artists=(legend, ),
            bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.savefig(FIGURES_DIR + filename)


def fix_legend(legend):
    for leg in legend.get_texts():
        text = leg.get_text()
        try:
            new_text = str(int(float(text)))
            leg.set_text(new_text)
        except ValueError:
            leg.set_text(LABEL_DICT[leg.get_text()])
        except:
            continue


#sns.set_palette('colorblind', n_colors=7)
def plot_cdf(values_labels_and_colors,
             xlabel,
             title=None,
             log=False,
             show=True,
             fname=None):
    #linestyles = cycle(['-', '--', '-.', ':'])
    for (_x_values, label, color) in values_labels_and_colors:
        x_values = sorted(_x_values)
        N = len(x_values)
        y_values = np.arange(N) / float(N)
        plt.plot(x_values, y_values, label=label, color=color)
        #linestyle=next(linestyles))
    if log:
        plt.xscale('log')
    xlim = plt.xlim()
    plt.xlim([0.0, xlim[-1]])
    plt.ylim([0.0, 1.0])
    if title:
        plt.title(title)
    legend = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel(xlabel)
    plt.ylabel('CDF')
    plt.grid(True)
    sns.despine()
    if fname:
        save_figure(fname, legend)
    if show:
        plt.show()


def get_theta_b(df):
    return df.query('K == 1')[['Cluster', 'ThetaUC']].groupby(
        'Cluster', as_index=False).aggregate(max)['ThetaUC']


###########
## PLOTS ##
###########
def f_u_plots(simdex_df, lemp_df, blocked_mm_df=None, models=NETFLIX_MODELS):
    simdex_rt = simdex_df.sort_values(by='comp_time').groupby(
        ['model', 'K'], as_index=False).first()[['model', 'K', 'comp_time']]
    simdex_rt['algo'] = 'SimDex'

    lemp_rt = lemp_df[['model', 'K', 'comp_time']]
    lemp_rt['algo'] = 'LEMP'

    if blocked_mm_df is not None:
        blocked_mm_rt = blocked_mm_df[['model', 'K', 'comp_time']]
        blocked_mm_rt['algo'] = 'Blocked Matrix Multiply'
        total_rt = pd.concat([simdex_rt, lemp_rt, blocked_mm_rt])
    else:
        total_rt = pd.concat([simdex_rt, lemp_rt])

    for model in models:
        data = total_rt.query('model == "%s"' % model)
        if len(data) == 0: continue
        max_runtime = data['comp_time'].max()
        sns.barplot(
            x='K',
            y='comp_time',
            hue='algo',
            data=data,
            linewidth=1.25,
            edgecolor='black')
        legend = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        start, end = plt.ylim()
        plt.title(model)
        plt.ylim([start, max_runtime * 1.1])
        plt.minorticks_on()
        plt.xlabel('K')
        plt.ylabel('Time (s)')
        sns.despine()
        save_figure('f-u-plot-%s' % model, legend)
        plt.show()


def num_clusters_vs_runtime(simdex_df, models):
    for model in models:
        table = simdex_df.query('model == "%s"' % model)
        if len(table) == 0: continue
        data = table.groupby(
            ['num_clusters', 'K'], as_index=False).aggregate(min)
        sns.barplot(
            x='K',
            y='comp_time',
            hue='num_clusters',
            data=data,
            linewidth=1.25,
            edgecolor='black')
        sns.barplot(
            x='K',
            y='cluster_time',
            hue='num_clusters',
            data=data,
            linewidth=1.25,
            edgecolor='black',
            color='#ffffff',
            hatch='x')

        legend = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        start, end = plt.ylim()
        plt.xlabel('K')
        plt.ylabel('Time (s)')
        plt.minorticks_on()
        sns.despine()
        plt.title(model)
        save_figure('runtime-vs-n-cluster-%s' % model, legend)
        plt.show()


def num_bins_vs_runtime(simdex_df, models):
    best_rt = simdex_df.sort_values(by='comp_time').groupby(
        ['model', 'K'], as_index=False).first()
    for model in models:
        best_rt_model = best_rt.query('model == "%s"' % model)
        if len(best_rt_model) == 0: continue
        data = []
        for _, row in best_rt_model.iterrows():
            K, num_clusters = row['K'], row['num_clusters']
            data_for_K = simdex_df.query(
                'model == "%s" and num_clusters == %d and K == %d' %
                (model, num_clusters, K))
            data.append(data_for_K)
        data = pd.concat(data)

        sns.barplot(
            x='K',
            y='comp_time',
            hue='num_bins',
            data=data,
            linewidth=1.25,
            edgecolor='black')

        legend = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        start, end = plt.ylim()
        plt.xlabel('K')
        plt.ylabel('Time (s)')
        plt.minorticks_on()
        sns.despine()
        plt.title(model)
        save_figure('runtime-vs-n-bins-%s' % model, legend)
        plt.show()

def reg_vs_runtime(simdex_df, blocked_mm_df, model_prefixes, regs):
    simdex_rt = simdex_df.sort_values(by='comp_time').groupby(
        ['model', 'K'], as_index=False).first()[['model', 'K', 'comp_time']]
    blocked_mm_rt = blocked_mm_df[['model', 'K', 'comp_time']]
    blocked_mm_rt['algo'] = 'Blocked Matrix Multiply'

    for model_prefix in model_prefixes:
        data = []
        for reg in regs:
            if reg == 'gold-standard':
                model = model_prefix + '-' + reg
                if 'Netflix' in model_prefix:
                    reg = '0.05'
                else:
                    reg = '1'
            else:
                model = model_prefix + '-reg-' + reg

            simdex_result = simdex_rt.query('model == "%s"' % model)
            simdex_result['reg'] = reg
            data.append(simdex_result)

            blocked_mm_result = blocked_mm_rt.query('model == "%s"' % model)
            blocked_mm_result['reg'] = 'Blocked Matrix Multiply'
            data.append(blocked_mm_result)

        data = pd.concat(data)

        if len(data) == 0: continue
        max_runtime = data['comp_time'].max()
        sns.barplot(
            x='K',
            y='comp_time',
            hue='reg',
            data=data,
            linewidth=1.25,
            edgecolor='black')
        legend = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        start, end = plt.ylim()
        plt.title(model_prefix)
        plt.ylim([start, max_runtime * 1.1])
        plt.minorticks_on()
        plt.xlabel('K')
        plt.ylabel('Time (s)')
        sns.despine()
        save_figure('reg-%s' % model_prefix, legend)
        plt.show()


def cdf(simdex_df,
        blocked_mm_df,
        models,
        K,
        column='theta_uc',
        num_clusters_to_plot=None,
        title=None,
        fname=None,
        csv_dir='items-visited'):
    best_simdex_rts = simdex_df.sort_values(by='comp_time').groupby(
        ['model', 'K', 'num_clusters'], as_index=False).first()

    #COLORS = cycle(['red', 'blue', 'orange', 'green', 'purple', 'black'])
    #COLORS = {
    #    256: 'red',
    #    512: 'blue',
    #    1024: 'orange',
    #    2048: 'green',
    #    4096: 'brown',
    #    8192: 'black'
    #}

    def get_vals(model):
        vals = []
        for filename in glob.iglob('%s/%s_items_visited_*' % (csv_dir, model)):
            df = pd.read_csv(filename)
            num_clusters = df['cluster_id'].max() + 1
            if num_clusters_to_plot and num_clusters not in num_clusters_to_plot:
                continue
            #if num_clusters != df['cluster_id'].max() + 1:
            #    continue
            theta_bs = df[column].fillna(0.0)
            simdex_rt = best_simdex_rts.query(
                'model == "%s" and K == %d and num_clusters == %d' %
                (model, K, num_clusters))['comp_time']
            if len(simdex_rt) == 0: continue
            blocked_mm_rt = blocked_mm_df.query('model == "%s" and K == %d' %
                                                (model, K))['comp_time']
            if len(blocked_mm_rt) == 0: continue
            #color = COLORS[
            #    num_clusters]  #next(COLORS)
            color = 'red' if blocked_mm_rt.values[0] < simdex_rt.values[0] else 'black'
            vals.append((theta_bs, '%s, C=%d' % (model, num_clusters), color))
        return vals

    vals = []
    for model in models:
        for val in get_vals(model):
            vals.append(val)
    if fname:
        fname = 'cdf-%s' % fname
    if title:
        title = title + ' ' + csv_dir
    plot_cdf(vals, column, title=title, log=False, fname=fname, show=True)


def theta_uc_items_visited_scatter(df, filename):
    sns.lmplot(
        'ThetaUC',
        'ItemsVisited',
        data=df.query('K == 1'),
        fit_reg=False,
        scatter_kws={'marker': 'X',
                     'alpha': 0.1})
    _, end = plt.xlim()
    plt.xlim([0, end])
    _, end = plt.ylim()
    plt.ylim([0, end])
    plt.title(filename)
    #save_figure(filename)
    plt.show()
