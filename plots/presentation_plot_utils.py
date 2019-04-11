from __future__ import division
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from plot_utils import LABEL_DICT, save_figure


def factor_analysis_poster(figsize=(8, 4)):
    #labels = 'Cluster', 'Index Construction', 'Cost Estimation', 'Index Traversal, Index Traversal w/out Blocking'

    # NOMAD Netflix 50 k = 1
    # NOMAD R2 50 k = 1

    # NETFLIX
    # w/o blocking
    times0 = [0.22, 0.11, 0.44, 43.63, 107.143]

    # R2
    # w/o blocking
    times1 = [1.12, .45, 1.509, 192.379, 267.40]

    y0 = times0
    y1 = times1

    fig, ax = plt.subplots(figsize=figsize)
    # Example data
    people = (
        r'Netflix, $f=50$',
        r'Yahoo R2, $f=50$',
        )
    y_pos = np.array([1.5, 3])

    H = 0.25

    y_pos_mid = [1.5 * H + y for y in y_pos]

    ax.barh(
        y_pos, [y0[0], y1[0]],
        linewidth=0.50,
        edgecolor='black',
        height=H,
        align='center',
        label='Clustering')
    y_pos = [H + y for y in y_pos]

    ax.barh(
        y_pos, [y0[1], y1[1]],
        linewidth=0.50,
        edgecolor='black',
        height=H,
        align='center',
        label='Index Construction')
    y_pos = [H + y for y in y_pos]

    #ax.barh(
    #    y_pos, [y0[2], y1[2]],
    #    linewidth=0.50,
    #    edgecolor='black',
    #    height=H,
    #    align='center',
    #    label='Cost Estimation')
    #y_pos = [H + y for y in y_pos]

    ax.barh(
        y_pos, [y0[3], y1[3]],
        linewidth=0.50,
        edgecolor='black',
        height=H,
        align='center',
        label='Index Traversal')
    y_pos = [H + y for y in y_pos]

    ax.barh(
        y_pos, [y0[4], y1[4]],
        linewidth=0.50,
        edgecolor='black',
        height=H,
        align='center',
        label='Index Traversal w/out Blocking')

    handles, labels = ax.get_legend_handles_labels()
    handles.reverse()
    labels.reverse()
    lgd = ax.legend(
        handles[::-1],
        labels[::-1],
        loc='upper center',
        bbox_to_anchor=(0, 0, 1, 1.3),
        ncol=2,
        frameon=False,
        columnspacing=None,
        labelspacing=None)

    ax.set_yticks(y_pos_mid)
    ax.set_yticklabels(people)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Time (seconds, log scale)')
    ax.set_xscale('log')
    ax.grid(True)

    sns.despine()
    save_figure('factor-analysis-poster', (lgd, ))


def benchmark_against_blocked_mm(
                    lemp_df,
                    fexipro_df,
                    blocked_mm_df,
                    model,
                    K=[1,50],
                    include_naive=True,
                    num_clusters=8,
                    batch_size=4096,
                    y_title=-0.35,
                    title=True,
                    log=False,
                    ncol=1,
                    bbox_to_anchor=(0.5, 1.25),
                    figsize=(10, 6.5)):

    fig, ax = plt.subplots(figsize=figsize)

    blocked_mm_rt = blocked_mm_df[['model', 'K', 'comp_time']]
    blocked_mm_rt['algo'] = 'Blocked MM'

    lemp_rt = lemp_df[['model', 'K', 'comp_time']]
    lemp_rt['algo'] = 'LEMP'

    fexipro_rt = fexipro_df[['model', 'K', 'comp_time']]
    fexipro_rt['algo'] = 'FEXIPRO'

    if include_naive:
        naive_rt = blocked_mm_df[['model', 'K', 'comp_time']]
        naive_rt['comp_time'] = blocked_mm_df['comp_time']*40
        naive_rt['algo'] = 'Naïve'

    k_clause = ' or '.join(['K == %d' % val for val in K])
    lemp_model_rt = lemp_rt.query('model == "%s" and (%s)' % (model, k_clause)).sort_values(by='K')
    fexipro_model_rt = fexipro_rt.query('model == "%s" and (%s)' % (model, k_clause)).sort_values(
        by='K')
    if include_naive:
        naive_model_rt = naive_rt.query('model == "%s" and (%s)' % (model, k_clause)).sort_values(
            by='K')
    blocked_mm_model_rt = blocked_mm_rt.query(
        'model == "%s" and (%s)' % (model, k_clause)).sort_values(by='K')

    print('Naive')
    print(naive_model_rt)

    print('LEMP')
    print(lemp_model_rt)

    print('FEXIPRO')
    print(fexipro_model_rt)

    print('Blocked MM')
    print(blocked_mm_model_rt)


    if include_naive:
        data = pd.concat([
            naive_model_rt, fexipro_model_rt, lemp_model_rt, blocked_mm_model_rt
        ])
    else:
        data = pd.concat([
            fexipro_model_rt, lemp_model_rt, blocked_mm_model_rt
        ])
    if len(data) == 0: return
    max_runtime = sorted([
        lemp_model_rt['comp_time'].max(), blocked_mm_model_rt['comp_time'].max(),
        fexipro_model_rt['comp_time'].max()
    ])[1]

    sns.barplot(
        ax=ax,
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
    if log:
        ax.set_yscale('log')
        ax.set_ylabel('Time for all users (s, log scale)')
    else:
        ax.set_ylabel('Time for all users (s)')
    plt.xlabel('K')

    plt.grid(True)
    if title:
        plt.title(LABEL_DICT[model] if model in LABEL_DICT else model, y=y_title)
    sns.despine()

    legend = plt.legend(loc='center', bbox_to_anchor=bbox_to_anchor, ncol=ncol)
    save_figure('benchmark-against-blocked-mm-%s' % model, (legend, ))
    plt.show()

def benchmark_against_naive(
                    lemp_df,
                    fexipro_df,
                    blocked_mm_df,
                    model,
                    num_clusters=8,
                    batch_size=4096,
                    bbox_to_anchor=(0.5, 1.25),
                    title=True,
                    ncol=3,
                    K=[1,50],
                    y_title=-0.35,
                    figsize=(10, 6.5)):

    lemp_rt = lemp_df[['model', 'K', 'comp_time']]
    lemp_rt['algo'] = 'LEMP'

    fexipro_rt = fexipro_df[['model', 'K', 'comp_time']]
    fexipro_rt['algo'] = 'FEXIPRO'

    naive_rt = blocked_mm_df[['model', 'K', 'comp_time']]
    naive_rt['comp_time'] = blocked_mm_df['comp_time']*40
    naive_rt['algo'] = 'Naïve'

    lemp_model_rt = lemp_rt.query('model == "%s"' % model).sort_values(by='K')
    fexipro_model_rt = fexipro_rt.query('model == "%s"' % model).sort_values(
        by='K')
    naive_model_rt = naive_rt.query('model == "%s"' % model).sort_values(
        by='K')


    data = pd.concat([
        naive_model_rt, fexipro_model_rt, lemp_model_rt
    ])
    if len(data) == 0: return
    max_runtime = sorted([
        lemp_model_rt['comp_time'].max(), fexipro_model_rt['comp_time'].max()
    ])[1]

    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(
        ax=ax,
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
    plt.ylabel('Time for all users (s)')
    plt.xlabel('K')

    plt.grid(True)
    if title:
        plt.title(LABEL_DICT[model] if model in LABEL_DICT else model, y=y_title)
    sns.despine()

    legend = plt.legend(loc='center', bbox_to_anchor=bbox_to_anchor, ncol=ncol)
    save_figure('benchmark-against-naive-%s' % model, (legend, ))
    plt.show()
