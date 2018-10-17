from __future__ import division
from math import ceil
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
import pandas as pd
import pylab
import os
import glob
import itertools
from plot_utils import LABEL_DICT, save_figure

def benchmark_against_blocked_mm(
                    lemp_df,
                    blocked_mm_df,
                    fexipro_df,
                    naive_df,
                    model,
                    num_clusters=1,
                    batch_size=4096,
                    y_title=-0.35):

    blocked_mm_rt = blocked_mm_df[['model', 'K', 'comp_time']]
    blocked_mm_rt['algo'] = 'Blocked MM'

    lemp_rt = lemp_df[['model', 'K', 'comp_time']]
    lemp_rt['algo'] = 'LEMP'

    fexipro_rt = fexipro_df[['model', 'K', 'comp_time']]
    fexipro_rt['algo'] = 'FEXIPRO'

    #% naive_rt = naive_df[['model', 'K', 'comp_time']]
    #% naive_rt['algo'] = 'Naive'

    lemp_model_rt = lemp_rt.query('model == "%s"' % model).sort_values(by='K')
    fexipro_model_rt = fexipro_rt.query('model == "%s"' % model).sort_values(
        by='K')
    # naive_model_rt = naive_rt.query('model == "%s"' % model).sort_values(
    #     by='K')
    blocked_mm_model_rt = blocked_mm_rt.query(
        'model == "%s"' % model).sort_values(by='K')


    data = pd.concat([
        blocked_mm_model_rt, lemp_model_rt, fexipro_model_rt, # naive_model_rt
    ])
    if len(data) == 0: return
    max_runtime = sorted([
        lemp_model_rt['comp_time'].max(), blocked_mm_model_rt['comp_time'].max(),
        fexipro_model_rt['comp_time'].max()
    ])[1]

    sns.barplot(
        x='K',
        y='comp_time',
        hue='algo',
        data=data,
        ci=None,
        linewidth=1.25,
        edgecolor='black')

    start, end = plt.ylim()
    # if not np.isnan(max_runtime):
    #     plt.ylim([start, max_runtime * 1.1])
    plt.minorticks_on()
    plt.ylabel('Time (s)')
    plt.xlabel('K')

    plt.grid(True)
    plt.title(LABEL_DICT[model] if model in LABEL_DICT else model, y=y_title)
    sns.despine()

    legend = plt.legend(loc='center', bbox_to_anchor=(0.5, 1.25), ncol=3)
    save_figure('benchmark-against-blocked-mm-%s' % model, (legend, ))
    plt.show()

def benchmark_against_naive(
                    lemp_df,
                    fexipro_df,
                    naive_df,
                    model,
                    num_clusters=1,
                    batch_size=4096,
                    K=[1,50],
                    y_title=-0.35):

    lemp_rt = lemp_df[['model', 'K', 'comp_time']]
    lemp_rt['algo'] = 'LEMP'

    fexipro_rt = fexipro_df[['model', 'K', 'comp_time']]
    fexipro_rt['algo'] = 'FEXIPRO'

    naive_rt = naive_df[['model', 'K', 'comp_time']]
    naive_rt['algo'] = 'Naive'

    lemp_model_rt = lemp_rt.query('model == "%s"' % model).sort_values(by='K')
    fexipro_model_rt = fexipro_rt.query('model == "%s"' % model).sort_values(
        by='K')
    naive_model_rt = naive_rt.query('model == "%s"' % model).sort_values(
        by='K')


    data = pd.concat([
        lemp_model_rt, fexipro_model_rt, naive_model_rt
    ])
    if len(data) == 0: return
    max_runtime = sorted([
        lemp_model_rt['comp_time'].max(), fexipro_model_rt['comp_time'].max()
    ])[1]

    sns.barplot(
        x='K',
        y='comp_time',
        hue='algo',
        data=data,
        linewidth=1.25,
        edgecolor='black')

    start, end = plt.ylim()
    # if not np.isnan(max_runtime):
    #     plt.ylim([start, max_runtime * 1.1])
    plt.minorticks_on()
    plt.ylabel('Time (s)')
    plt.xlabel('K')

    plt.grid(True)
    plt.title(LABEL_DICT[model] if model in LABEL_DICT else model, y=y_title)
    sns.despine()

    legend = plt.legend(loc='center', bbox_to_anchor=(0.5, 1.25), ncol=3)
    save_figure('benchmark-against-naive-%s' % model, (legend, ))
    plt.show()
