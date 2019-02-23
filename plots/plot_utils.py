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

FIGURES_DIR = 'figures/'
if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)
APPETIZER_LABEL_DICT = {
    'nomad-Netflix-50-reg-0.05': r'Netflix, $f=50$',
    'nomad-R2-50-reg-0.000001': r'Yahoo R2, $f=50$',
        }

LABEL_DICT = {
    'bpr-Netflix-10-reg-0.00001': r'Netflix-BPR, $f=10$',
    'bpr-Netflix-25-reg-0.0001': r'Netflix-BPR, $f=25$',
    'bpr-Netflix-50-reg-0.0001': r'Netflix-BPR, $f=50$',
    'bpr-Netflix-100-reg-0.0001': r'Netflix-BPR, $f=100$',
    'fexipro-paper-Netflix-50': r'''Netflix-libPMF, $f=50$''',
    'fexipro-paper-KDD-50': r'''KDD-libPMF, $f=50$''',
    'lemp-paper-Netflix-noav-10': r'''Netflix-DSGD, $f=10$''',
    'nomad-Netflix-10': r'Netflix-DSGD, $f=10$',
    'nomad-Netflix-10-reg-0.05': r'Netflix-NOMAD, $f=10$',
    'nomad-Netflix-25': r'Netflix-NOMAD, $f=25$',
    'nomad-Netflix-25-reg-0.05': r'Netflix-NOMAD, $f=25$',
    'lemp-paper-Glove-50': r'GloVe Twitter, $f=50$',
    'lemp-paper-Glove-100': r'GloVe Twitter, $f=100$',
    'lemp-paper-Glove-200': r'GloVe Twitter, $f=200$',
    'lemp-paper-Netflix-50': r'''Netflix-DSGD (avg), $f=50$''',
    'lemp-paper-Netflix-noav-50': r'''Netflix-DSGD, $f=50$''',
    'nomad-Netflix-50': r'Netflix-NOMAD, $f=50$',
    'nomad-Netflix-50-reg-0.05': r'Netflix-NOMAD, $f=50$',
    'lemp-paper-Netflix-noav-100': r'''Netflix-DSGD, $f=100$''',
    'nomad-Netflix-100': r'Netflix-NOMAD, $f=100$',
    'nomad-Netflix-100-reg-0.05': r'Netflix-NOMAD, $f=100$',
    'nomad-KDD-10-reg-1': r'KDD-NOMAD, $f=10$',
    'nomad-KDD-25-reg-0.001': r'KDD-NOMAD, $f=25$',
    'nomad-KDD-50-reg-1': r'KDD-NOMAD, $f=50$',
    'lemp-paper-KDD-50': r'KDD-REF, $f=51$',
    'nomad-KDD-100-reg-1': r'KDD-NOMAD, $f=100$',
    'nomad-R2-10': r'R2-NOMAD, $f=10$',
    'nomad-R2-10-reg-0.001': r'R2-NOMAD, $f=10$',
    'nomad-R2-25': r'R2-NOMAD, $f=25$',
    'nomad-R2-25-reg-0.001': r'R2-NOMAD, $f=25$',
    'nomad-R2-50': r'R2-NOMAD, $f=50$',
    'nomad-R2-50-reg-0.001': r'R2-NOMAD-bad, $f=50$',
    'nomad-R2-50-reg-0.000001': r'R2-NOMAD, $f=50$',
    'nomad-R2-100': r'R2-NOMAD, $f=100$',
    'nomad-R2-100-reg-0': r'R2-NOMAD, $f=100$',
}

BLOG_POST_LABEL_DICT = {
    'nomad-Netflix-10-reg-0.05': r'Netflix Prize, $f=10$',
    'nomad-Netflix-25': r'Netflix Prize, $f=25$',
    'nomad-Netflix-25-reg-0.05': r'Netflix Prize, $f=25$',
    'nomad-Netflix-50': r'Netflix Prize, $f=50$',
    'nomad-Netflix-50-reg-0.05': r'Netflix Prize, $f=50$',
    'nomad-Netflix-100': r'Netflix Prize, $f=100$',
    'nomad-Netflix-100-reg-0.05': r'Netflix Prize, $f=100$',
    'nomad-KDD-10-reg-1': r'Yahoo Music KDD, $f=10$',
    'nomad-KDD-25-reg-0.001': r'Yahoo Music KDD, $f=25$',
    'nomad-KDD-50-reg-1': r'Yahoo Music KDD, $f=50$',
    'nomad-KDD-100-reg-1': r'Yahoo Music KDD, $f=100$',
    'nomad-R2-10': r'Yahoo Music R2, $f=10$',
    'nomad-R2-10-reg-0.001': r'Yahoo Music R2, $f=10$',
    'nomad-R2-25': r'Yahoo Music R2, $f=25$',
    'nomad-R2-25-reg-0.001': r'Yahoo Music R2, $f=25$',
    'nomad-R2-50': r'Yahoo Music R2, $f=50$',
    'nomad-R2-50-reg-0.001': r'Yahoo Music R2, $f=50$',
    'nomad-R2-100': r'Yahoo Music R2, $f=100$',
    'nomad-R2-100-reg-0.01': r'Yahoo Music R2, $f=100$',
}

HATCHES = ['0', '-', '/', 'x', '.', '|', '+', '//', '\\', 'o', '*']


###########
## UTILS ##
###########
def reverse_palette(palette_str, num_colors):
    palette = sns.color_palette(palette_str, num_colors)
    palette.reverse()
    return palette


def save_figure(filename, extra_artists=None, tight=True):
    filename = filename.replace('.', '-')
    filename += '.pdf'
    if extra_artists:
        plt.savefig(
            FIGURES_DIR + filename,
            bbox_extra_artists=extra_artists,
            bbox_inches='tight')
    else:
        if tight:
            plt.tight_layout()
        plt.savefig(FIGURES_DIR + filename)


def replace_legend_labels(legend):
    for leg in legend.get_texts():
        text = leg.get_text()
        leg.set_text(LABEL_DICT[text])


def modify_legend(legend, legend_pref_suf):
    if len(legend_pref_suf) == 1:
        for i, leg in enumerate(legend.get_texts()):
            text = leg.get_text()
            prefix, suffix = legend_pref_suf[0]
            leg.set_text(prefix + text + suffix)
    else:
        for i, leg in enumerate(legend.get_texts()):
            text = leg.get_text()
            prefix, suffix = legend_pref_suf[i]
            leg.set_text(prefix + text + suffix)


def add_hatches(bar, num_groups):
    for i, single_bar in enumerate(bar.patches):
        # Set a different hatch for each bar
        single_bar.set_hatch(HATCHES[int(i / num_groups)])


def plot_cdf(values_labels_and_lines,
             xlabel,
             x_line=None,
             y_line=None,
             title=None,
             log=False,
             show=True,
             fname=None):
    if x_line:
        plt.axvline(x=x_line, color='black', linestyle='-', linewidth=2.5)
    if y_line:
        plt.axhline(y=y_line, color='black', linestyle='-', linewidth=2.5)
    for (_x_values, label, line_style) in values_labels_and_lines:
        x_values = sorted(_x_values)
        N = len(x_values)
        y_values = np.arange(N) / float(N)
        plt.plot(x_values, y_values, label=label, linestyle=line_style)
    if log:
        plt.xscale('log')
    xlim = plt.xlim()
    plt.xlim([0.0, xlim[-1]])
    yticks = np.arange(0.0, 1.1, 0.1)
    plt.yticks(yticks)
    plt.ylim([0.0, 1.0])
    plt.minorticks_on()
    if title:
        plt.title(title, y=1.05)
    legend = plt.legend(
        bbox_to_anchor=(0.5, -0.19),
        loc='upper center',
        borderaxespad=0.,
        frameon=True,
        edgecolor='black',
        framealpha=0.6)
    replace_legend_labels(legend)
    plt.xlabel(xlabel)
    plt.ylabel('CDF')
    plt.grid(True)
    sns.despine()
    if fname:
        save_figure(fname, (legend))
    if show:
        plt.show()


###########
## PLOTS ##
###########
def runtime_estimates_plot(models,
                           lemp_estimates_df,
                           lemp_truth_df,
                           fexipro_estimates_df,
                           fexipro_truth_df,
                           simdex_estimates_df,
                           simdex_truth_df,
                           blocked_mm_estimates_df,
                           blocked_mm_truth_df,
                           linestyle='--',
                           markerstyle='o',
                           figsize=(15, 8)):
    def fexipro_truth_query_fn(row):
        return 'model == "%s" and K == %d and alg == "%s" and scaling_value == %f and sigma == %f' % (
            row['model'], row['K'], row['alg'], row['scaling_value'],
            row['sigma'])

    def lemp_truth_query_fn(row):
        return 'model == "%s" and K == %d' % (row['model'], row['K'])

    def simdex_truth_query_fn(row):
        return 'model == "%s" and K == %d and num_clusters == %d and batch_size == %d' % (
            row['model'], row['K'], row['num_clusters'], row['batch_size'])

    def blocked_mm_truth_query_fn(row):
        return 'model == "%s" and K == %d' % (row['model'], row['K'])

    def runtimes_for_index_type(model, estimates_df, truth_df, sample_time_col,
                                truth_query_fn):
        L2_CACHE_SIZE = 256000
        MAX_MEM_SIZE = 257840 * 1024 * 1024

        ratios, estimate_rts, true_rts, error_rts = [], [], [], []
        for user_sample_ratio in estimates_df['user_sample_ratio'].unique():

            num_users = 480189 if 'Netflix' in model else 1000990 if 'KDD' in model else 1823179
            num_items = 17770 if 'Netflix' in model else 624961 if 'KDD' in model else 136736

            estimates_for_ratio = estimates_df.query(
                'user_sample_ratio == %f' % user_sample_ratio)
            row = estimates_for_ratio.iloc[0]
            num_latent_factors = row['num_latent_factors']

            if user_sample_ratio == 0.0:
                num_users_per_block = 4 * L2_CACHE_SIZE / (
                    8 * num_latent_factors)
                while (num_users_per_block * num_items * 8 > MAX_MEM_SIZE):
                    num_users_per_block /= 2
                user_sample_ratio = num_users_per_block / num_users

            estimate_rt = estimates_for_ratio[
                sample_time_col].mean() * num_users
            error_rt = estimates_for_ratio[sample_time_col].std() * num_users
            true_rt = truth_df.query(truth_query_fn(row))['comp_time'].min()
            ratios.append(user_sample_ratio)
            estimate_rts.append(estimate_rt)
            true_rts.append(true_rt)
            error_rts.append(error_rt)
        ratios = np.array(ratios) * 100
        x = list(zip(ratios, estimate_rts, true_rts, error_rts))
        x = sorted(x, key=lambda v: v[0])
        return zip(*x)

    fig, ax_arr = plt.subplots(nrows=1, ncols=len(models), figsize=figsize)
    ax_arr = np.ravel(ax_arr)
    for i, model_to_plot in enumerate(models):
        ratios, lemp_model_estimates, lemp_model_truth, lemp_model_errors = runtimes_for_index_type(
            model_to_plot,
            lemp_estimates_df.query('model == "%s"' % model_to_plot),
            lemp_truth_df.query('model == "%s"' % model_to_plot),
            'lemp_sample_time', lemp_truth_query_fn)

        ax_arr[i].errorbar(
            ratios,
            lemp_model_estimates,
            yerr=lemp_model_errors,
            label='LEMP estimate',
            linestyle=linestyle,
            marker=markerstyle)
        ax_arr[i].plot(ratios, lemp_model_truth, label='LEMP')

        ratios, fexipro_si_model_estimates, fexipro_si_model_truth, fexipro_si_errors = runtimes_for_index_type(
            model_to_plot,
            fexipro_estimates_df.query(
                'model == "%s" and alg == "SI"' % model_to_plot),
            fexipro_truth_df.query(
                'model == "%s" and alg == "SI"' % model_to_plot),
            'fexipro_sample_time', fexipro_truth_query_fn)

        ax_arr[i].errorbar(
            ratios,
            fexipro_si_model_estimates,
            yerr=fexipro_si_errors,
            label='FEXIPRO-SI estimate',
            linestyle=linestyle,
            marker=markerstyle)
        ax_arr[i].plot(ratios, fexipro_si_model_truth, label='FEXIPRO-SI')

        ratios, fexipro_sir_model_estimates, fexipro_sir_model_truth, fexipro_sir_errors = runtimes_for_index_type(
            model_to_plot,
            fexipro_estimates_df.query(
                'model == "%s" and alg == "SIR"' % model_to_plot),
            fexipro_truth_df.query(
                'model == "%s" and alg == "SIR"' % model_to_plot),
            'fexipro_sample_time', fexipro_truth_query_fn)

        ax_arr[i].errorbar(
            ratios,
            fexipro_sir_model_estimates,
            yerr=fexipro_sir_errors,
            label='FEXIPRO-SIR estimate',
            linestyle=linestyle,
            marker=markerstyle)
        ax_arr[i].plot(ratios, fexipro_sir_model_truth, label='FEXIPRO-SIR')

        ratios, simdex_model_estimates, simdex_model_truth, simdex_model_errors = runtimes_for_index_type(
            model_to_plot,
            simdex_estimates_df.query('model == "%s"' % model_to_plot),
            simdex_truth_df.query('model == "%s"' % model_to_plot),
            'simdex_sample_time', simdex_truth_query_fn)

        ax_arr[i].errorbar(
            ratios,
            simdex_model_estimates,
            yerr=simdex_model_errors,
            label='Maximus estimate',
            linestyle=linestyle,
            marker=markerstyle)
        ax_arr[i].plot(ratios, simdex_model_truth, label='Maximus')

        ratios, blocked_mm_model_estimates, blocked_mm_model_truth, blocked_mm_errors = runtimes_for_index_type(
            model_to_plot,
            blocked_mm_estimates_df.query('model == "%s"' % model_to_plot),
            blocked_mm_truth_df.query('model == "%s"' % model_to_plot),
            'blocked_mm_sample_time', blocked_mm_truth_query_fn)

        ax_arr[i].errorbar(
            ratios,
            blocked_mm_model_estimates,
            yerr=blocked_mm_errors,
            label='Blocked MM estimate',
            linestyle=linestyle,
            marker=markerstyle)
        ax_arr[i].plot(ratios, blocked_mm_model_truth, label='Blocked MM')

        ax_arr[i].set_yscale('log')
        handles, labels = ax_arr[i].get_legend_handles_labels()
        handles = handles[:int(len(handles) / 2)]
        labels = labels[:int(len(labels) / 2)]
        legend = ax_arr[i].legend(
            handles,
            labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.37),
            ncol=int(len(handles) / 2) + 1,
            columnspacing=1.7,
            labelspacing=0.2)
        ax_arr[i].set_ylabel('Time (s), log scale', labelpad=15)
        ax_arr[i].set_xlabel('User sample percentage', labelpad=15)
        ax_arr[i].set_xlim([0, 1.02])

        fmt = '%.1f%%'  # Format you want the ticks, e.g. '40%'
        xticks = mtick.FormatStrFormatter(fmt)
        ax_arr[i].xaxis.set_major_formatter(xticks)
        ax_arr[i].tick_params(axis='x', which='major', pad=15)

        ax_arr[i].grid(True)
        sns.despine()
        save_figure('runtime-estimate', (legend, ))


def benchmark_against_blocked_mm_multi(lemp_df,
                                       blocked_mm_df,
                                       fexipro_df,
                                       models,
                                       num_clusters=1,
                                       batch_size=4096,
                                       bbox_to_anchor=(0, 0, 1, 1),
                                       y_title=-0.35,
                                       figsize=(10, 5),
                                       include_naive=False):

    fig, ax_arr = plt.subplots(1, len(models), figsize=figsize)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.3)

    for i, model in enumerate(models):

        blocked_mm_rt = blocked_mm_df[['model', 'K', 'comp_time']]
        blocked_mm_rt['algo'] = 'Blocked MM (Brute Force)'

        lemp_rt = lemp_df[['model', 'K', 'comp_time']]
        lemp_rt['algo'] = 'LEMP'

        fexipro_rt = fexipro_df[['model', 'K', 'comp_time']]
        fexipro_rt['algo'] = 'FEXIPRO'


        lemp_model_rt = lemp_rt.query('model == "%s"' % model).sort_values(
            by='K')
        fexipro_model_rt = fexipro_rt.query(
            'model == "%s"' % model).sort_values(by='K')
        blocked_mm_model_rt = blocked_mm_rt.query(
            'model == "%s"' % model).sort_values(by='K')

        if include_naive:
            naive_rt = blocked_mm_df[['model', 'K']]
            naive_rt['comp_time'] = blocked_mm_df['comp_time'] * 40
            naive_rt['algo'] = 'Naive'
            naive_model_rt = naive_rt.query('model == "%s"' % model).sort_values(
                 by='K')
            data = pd.concat([
                blocked_mm_model_rt,
                lemp_model_rt,
                fexipro_model_rt,
                naive_model_rt
            ])
        else:
            data = pd.concat([
                blocked_mm_model_rt,
                lemp_model_rt,
                fexipro_model_rt,
            ])
        if len(data) == 0: return
        max_runtime = sorted([
            lemp_model_rt['comp_time'].max(),
            blocked_mm_model_rt['comp_time'].max(),
            fexipro_model_rt['comp_time'].max()
        ])[1]

        sns.barplot(
            x='K',
            y='comp_time',
            hue='algo',
            data=data,
            ci=None,
            ax=ax_arr[i],
            linewidth=1.25,
            edgecolor='black')
        ax_arr[i].legend_.remove()
        ax_arr[i].grid(True)
        ax_arr[i].set_ylabel('Time (s)')
        ax_arr[i].set_xlabel('K')
        ax_arr[i].set_title(
            APPETIZER_LABEL_DICT[model] if model in LABEL_DICT else model, y=y_title)

    sns.despine()
    legend = ax_arr[0].legend(
        loc='upper center',
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=plt.gcf().transFigure,
        ncol=3)

    save_figure('benchmark-against-blocked-mm-appetizer', (legend, ))


def blocked_mm_lemp_fexipro_plot(blocked_mm_df,
                                 lemp_df,
                                 fexipro_df,
                                 fexipro_si_df,
                                 model,
                                 title,
                                 y_title=-0.35,
                                 figsize=(8, 8)):
    plt.figure(figsize=figsize)

    blocked_mm_rt = blocked_mm_df[['model', 'K', 'comp_time']]
    blocked_mm_rt['algo'] = 'Blocked MM'

    lemp_rt = lemp_df[['model', 'K', 'comp_time']]
    lemp_rt['algo'] = 'LEMP'

    fexipro_rt = fexipro_df[['model', 'K', 'comp_time']]
    fexipro_rt['algo'] = 'FEXIPRO'

    fexipro_si_rt = fexipro_si_df[['model', 'K', 'comp_time']]
    fexipro_si_rt['algo'] = 'FEXIPRO-SI'

    lemp_model_rt = lemp_rt.query('model == "%s"' % model).sort_values(by='K')
    fexipro_model_rt = fexipro_rt.query('model == "%s"' % model).sort_values(
        by='K')
    fexipro_si_model_rt = fexipro_si_rt.query(
        'model == "%s"' % model).sort_values(by='K')
    blocked_mm_model_rt = blocked_mm_rt.query(
        'model == "%s"' % model).sort_values(by='K')

    data = pd.concat([
        blocked_mm_model_rt, lemp_model_rt, fexipro_model_rt,
        fexipro_si_model_rt
    ])
    if len(data) == 0: return

    max_runtime = max([
        lemp_model_rt['comp_time'].max(),
        blocked_mm_model_rt['comp_time'].max(),
        fexipro_model_rt['comp_time'].max(),
        fexipro_si_model_rt['comp_time'].max(),
    ])

    sns.barplot(
        x='K',
        y='comp_time',
        hue='algo',
        data=data,
        linewidth=1.25,
        edgecolor='black')

    start, end = plt.ylim()
    if not np.isnan(max_runtime):
        plt.ylim([start, max_runtime * 1.1])
    plt.minorticks_on()
    plt.ylabel('Time (s)')
    plt.xlabel('K')

    plt.grid(True)
    plt.title(title, y=y_title)
    sns.despine()

    legend = plt.legend(loc='2', bbox_to_anchor=(1, 1.05))
    save_figure('blocked_mm-lemp-fexipro', (legend, ))
    plt.show()


def factor_analysis(figsize=(8, 4)):
    labels = 'Cluster', 'Index Construction', 'Cost Estimation', 'Index Traversal'

    # NOMAD Netflix 50 k = 1
    # NOMAD R2 50 k = 1

    # NETFLIX
    # w/o blocking
    times0 = [0.22, 0.11, 0.44, 107.143]
    # w/ blocking
    times1 = [0.24, 0.11, 0.44, 43.63]

    # R2
    # w/o blocking
    times2 = [1.12, .45, 1.509, 267.40]
    # w/o blocking
    times3 = [1.12, .45, 1.509, 192.379]

    y0 = times0
    y1 = times1
    y2 = times2
    y3 = times3

    fig, ax = plt.subplots(figsize=figsize)
    # Example data
    people = (
        'Netflix w/o\nItem Blocking',
        'Netflix with\nItem Blocking',
        'R2 w/o\nItem Blocking',
        'R2 with\nItem Blocking', )
    y_pos = np.array([0, 1.5, 3, 4.5])

    H = 0.25

    y_pos_mid = [1.5 * H + y for y in y_pos]

    ax.barh(
        y_pos, [y0[0], y1[0], y2[0], y3[0]],
        linewidth=0.50,
        edgecolor='black',
        height=H,
        align='center',
        label='Clustering')
    y_pos = [H + y for y in y_pos]

    ax.barh(
        y_pos, [y0[1], y1[1], y2[1], y3[1]],
        linewidth=0.50,
        edgecolor='black',
        height=H,
        align='center',
        label='Index Construction')
    y_pos = [H + y for y in y_pos]

    ax.barh(
        y_pos, [y0[2], y1[2], y2[2], y3[2]],
        linewidth=0.50,
        edgecolor='black',
        height=H,
        align='center',
        label='Cost Estimation')
    y_pos = [H + y for y in y_pos]

    ax.barh(
        y_pos, [y0[3], y1[3], y2[3], y3[3]],
        linewidth=0.50,
        edgecolor='black',
        height=H,
        align='center',
        label='Index Traversal')

    handles, labels = ax.get_legend_handles_labels()
    handles.reverse()
    labels.reverse()
    lgd = ax.legend(
        handles[::-1],
        labels[::-1],
        loc='upper center',
        bbox_to_anchor=(0, 0, 1, 1.35),
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
    save_figure('factor-analysis', (lgd, ))


def f_u_plot_single(simdex_df,
                    lemp_df,
                    blocked_mm_df,
                    fexipro_df,
                    fexipro_si_df,
                    sampling_df,
                    model,
                    num_clusters=1,
                    batch_size=4096,
                    y_title=-0.35):

    simdex_df = simdex_df.query('num_clusters == %d and batch_size == %d' %
                                (num_clusters, batch_size))

    both_df = pd.concat([
        simdex_df[['model', 'K', 'comp_time']], blocked_mm_df[[
            'model', 'K', 'comp_time'
        ]].query('model != "nomad-Netflix-25-reg-0.05"')
    ])
    both_rt = both_df.sort_values(by='comp_time').groupby(
        ['model', 'K'], as_index=False).first()
    both_rt['algo'] = 'SimDex'

    simdex_rt = simdex_df.sort_values(by='comp_time').groupby(
        ['model', 'K'], as_index=False).first()
    simdex_rt['algo'] = 'SimDex-Index Only'

    blocked_mm_rt = blocked_mm_df[['model', 'K', 'comp_time']]
    blocked_mm_rt['algo'] = 'Blocked MM Only'

    lemp_rt = lemp_df[['model', 'K', 'comp_time']]
    lemp_rt['algo'] = 'LEMP'

    fexipro_rt = fexipro_df[['model', 'K', 'comp_time']]
    fexipro_rt['algo'] = 'FEXIPRO'

    fexipro_si_rt = fexipro_si_df[['model', 'K', 'comp_time']]
    fexipro_si_rt['algo'] = 'FEXIPRO-SI'

    both_model_rt = both_rt.query('model == "%s"' % model).sort_values(by='K')
    lemp_model_rt = lemp_rt.query('model == "%s"' % model).sort_values(by='K')
    fexipro_model_rt = fexipro_rt.query('model == "%s"' % model).sort_values(
        by='K')
    fexipro_si_model_rt = fexipro_si_rt.query(
        'model == "%s"' % model).sort_values(by='K')
    simdex_model_rt = simdex_rt.query('model == "%s"' % model).sort_values(
        by='K')
    blocked_mm_model_rt = blocked_mm_rt.query(
        'model == "%s"' % model).sort_values(by='K')
    sampling_model_rt = sampling_df.query('model == "%s"' % model).sort_values(
        by='K')

    if len(sampling_model_rt) > 0:
        both_model_rt['comp_time'] += sampling_model_rt['comp_time'].values

    data = pd.concat([
        both_model_rt, blocked_mm_model_rt, simdex_model_rt, lemp_model_rt,
        fexipro_model_rt, fexipro_si_model_rt
    ])
    if len(data) == 0: return
    max_runtime = sorted([
        lemp_model_rt['comp_time'].max(), simdex_model_rt['comp_time'].max(),
        blocked_mm_model_rt['comp_time'].max(),
        fexipro_model_rt['comp_time'].max(),
        fexipro_si_model_rt['comp_time'].max()
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

    legend = plt.legend(loc='2', bbox_to_anchor=(1, 1.05))
    save_figure('f-u-plot-single', (legend, ))
    plt.show()


def choose_runtimes_from_optimizer(simdex_df, blocked_mm_df, decision_rule_df):
    df_dict = {'model': [], 'K': [], 'comp_time': []}
    for _, row in decision_rule_df.iterrows():
        model, K = row['model'], row['K']
        df_dict['model'].append(model)
        df_dict['K'].append(K)

        simdex_rt = simdex_df.query('model == "%s" and K == %d' %
                                    (model, K))['comp_time'].min()
        blocked_mm_rt = blocked_mm_df.query('model == "%s" and K == %d' %
                                            (model, K))['comp_time'].min()
        overhead = row['blocked_mm_sample_time']

        comp_time = simdex_rt + overhead if row[
            'simdex_wins'] else blocked_mm_rt
        df_dict['comp_time'].append(comp_time)
    return pd.DataFrame.from_dict(df_dict)


def f_u_plots(simdex_df,
              lemp_df,
              blocked_mm_df,
              fexipro_df,
              fexipro_si_df,
              sampling_df,
              models,
              num_clusters=8,
              batch_size=4096,
              nrows=4,
              figsize=(28, 28),
              y_title=1.05,
              bbox_to_anchor=(0, 0, 1, 1)):
    legend = None  # save for later
    # if len(models) == 5:
    #     fig = plt.figure(figsize=(22, 10))
    #     gs = gridspec.GridSpec(2, 6)

    #     ax1 = plt.subplot(gs[0, 0:2])
    #     ax2 = plt.subplot(gs[0, 2:4])
    #     ax3 = plt.subplot(gs[0, 4:6])
    #     ax4 = plt.subplot(gs[1, 1:3])
    #     ax5 = plt.subplot(gs[1, 3:5])

    #     ax_arr = [ax1,ax2,ax3,ax4,ax5]
    # else:
    ncols = int(ceil(len(models) / nrows))
    fig, ax_arr = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    num_axes = nrows * ncols
    i = 0
    # center last row
    ax_arr = np.ravel(ax_arr)
    while len(models) < num_axes:
        fig.delaxes(ax_arr[(nrows - 1) * ncols + ncols - 1 - i])
        fig.delaxes(ax_arr[(nrows - 1) * ncols + i])
        # append deleted axis at the end
        temp = ax_arr[(nrows - 1) * ncols + i]
        ax_arr = np.delete(ax_arr, (nrows - 1) * ncols + i)
        ax_arr = np.append(ax_arr, temp)
        i += 1
        num_axes -= 2

    #_simdex_df = simdex_df.sort_values(by='comp_time').groupby(
    #    ['model', 'K'], as_index=False).first()
    _simdex_df = simdex_df.query('num_clusters == %d and batch_size == %d' %
                                 (num_clusters, batch_size))

    both_df = choose_runtimes_from_optimizer(_simdex_df, blocked_mm_df,
                                             sampling_df)
    both_rt = both_df.sort_values(by='comp_time').groupby(
        ['model', 'K'], as_index=False).first()
    both_rt['algo'] = 'Maximus + Blocked MM + Optimizer'

    simdex_rt = _simdex_df.sort_values(by='comp_time').groupby(
        ['model', 'K'], as_index=False).first()
    simdex_rt['algo'] = 'Maximus'

    blocked_mm_rt = blocked_mm_df[['model', 'K', 'comp_time']]
    blocked_mm_rt['algo'] = 'Blocked MM'

    lemp_rt = lemp_df[['model', 'K', 'comp_time']]
    lemp_rt['algo'] = 'LEMP'

    fexipro_rt = fexipro_df[['model', 'K', 'comp_time']]
    fexipro_rt['algo'] = 'FEXIPRO-SIR'

    fexipro_si_rt = fexipro_si_df[['model', 'K', 'comp_time']]
    fexipro_si_rt['algo'] = 'FEXIPRO-SI'

    all_speedups = []
    simdex_vs_lemp = []
    all_fexipro_speedups = []
    all_fexipro_si_speedups = []
    all_percent_overheads = []
    all_overheads = []

    for i, model in enumerate(models):
        both_model_rt = both_rt.query('model == "%s"' % model).groupby(
            ['model', 'K'], as_index=False).first().sort_values(by='K')
        lemp_model_rt = lemp_rt.query('model == "%s"' % model).groupby(
            ['model', 'K'], as_index=False).first().sort_values(by='K')
        fexipro_model_rt = fexipro_rt.query('model == "%s"' % model).groupby(
            ['model', 'K'], as_index=False).first().sort_values(by='K')
        fexipro_si_model_rt = fexipro_si_rt.query(
            'model == "%s"' % model).groupby(
                ['model', 'K'], as_index=False).first().sort_values(by='K')
        simdex_model_rt = simdex_rt.query('model == "%s"' % model).groupby(
            ['model', 'K'], as_index=False).first().sort_values(by='K')
        blocked_mm_model_rt = blocked_mm_rt.query(
            'model == "%s"' % model).groupby(
                ['model', 'K'], as_index=False).first().sort_values(by='K')
        sampling_model_rt = sampling_df.query('model == "%s"' % model).groupby(
            ['model', 'K'], as_index=False).first().sort_values(by='K')

        # add sampling overhead
        overheads = sampling_model_rt['blocked_mm_sample_time'].values
        all_overheads.append(overheads)
        percent_overheads = overheads / (
            both_model_rt['comp_time'].min() + overheads)
        all_percent_overheads.append(percent_overheads)
        print(model, percent_overheads)
        for _, row in simdex_model_rt[['K', 'batch_size',
                                       'num_clusters']].iterrows():
            K, batch_size, num_clusters = row[0], row[1], row[2]
            print(K, batch_size, num_clusters)

        speed_ups = lemp_model_rt['comp_time'].values / both_model_rt[
            'comp_time'].values
        all_speedups.append(speed_ups)

        simdex_vs_lemp_speed_ups = lemp_model_rt[
            'comp_time'].values / simdex_model_rt['comp_time'].values
        simdex_vs_lemp.append(simdex_vs_lemp_speed_ups)

        fexipro_speed_ups = fexipro_model_rt[
            'comp_time'].values / both_model_rt['comp_time'].values
        all_fexipro_speedups.append(fexipro_speed_ups)

        fexipro_si_speed_ups = fexipro_si_model_rt[
            'comp_time'].values / both_model_rt['comp_time'].values
        all_fexipro_si_speedups.append(fexipro_si_speed_ups)

        data = pd.concat([
            blocked_mm_model_rt, simdex_model_rt, lemp_model_rt,
            fexipro_model_rt, fexipro_si_model_rt
        ])
        if len(data) == 0:
            continue
        max_runtime = sorted([
            lemp_model_rt['comp_time'].max(),
            simdex_model_rt['comp_time'].max(),
            blocked_mm_model_rt['comp_time'].max(),
            fexipro_model_rt['comp_time'].max(),
            fexipro_si_model_rt['comp_time'].max()
        ])[3]

        sns.barplot(
            x='K',
            y='comp_time',
            hue='algo',
            data=data,
            linewidth=1.25,
            edgecolor='black',
            ax=ax_arr[i],
            ci=None)
        #add_hatches(bar, num_groups=4)

        ax_arr[i].legend_.remove()
        #legend = ax_arr[i].legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0.)

        start, end = ax_arr[i].get_ylim()
        ax_arr[i].set_ylim([start, max_runtime * 1.1])
        ax_arr[i].minorticks_on()
        if i % nrows == 0:
            ax_arr[i].set_ylabel('Time (s)')
        else:
            ax_arr[i].set_ylabel('')
        if int(i / ncols) == nrows - 1 or int(i / ncols) == nrows - 2 and (
                i % ncols == 0 or i % ncols == ncols - 1):
            ax_arr[i].set_xlabel('K')
        else:
            ax_arr[i].set_xlabel('')

        ax_arr[i].grid(True)
        ax_arr[i].set_title(LABEL_DICT[model], y=y_title)
        sns.despine()

    all_speedups = np.ravel(all_speedups)
    print('Average LEMP speedup: ' + str(np.mean(all_speedups)))
    print('Min LEMP speedup: ' + str(np.min(all_speedups)))
    print('Max LEMP speedup: ' + str(np.max(all_speedups)))

    simdex_vs_lemp = np.ravel(simdex_vs_lemp)
    print('Percent of time SimDex is better than LEMP: ' +
          str(np.mean(simdex_vs_lemp > 1.0)))

    all_fexipro_speedups = np.ravel(all_fexipro_speedups)
    print('Average FEXIPRO-SIR speedup: ' + str(np.mean(all_fexipro_speedups)))
    print('Min FEXIPRO-SIR speedup: ' + str(np.min(all_fexipro_speedups)))
    print('Max FEXIPRO-SIR speedup: ' + str(np.max(all_fexipro_speedups)))

    all_fexipro_si_speedups = np.ravel(all_fexipro_si_speedups)
    print('Average FEXIPRO-SI speedup: ' +
          str(np.mean(all_fexipro_si_speedups)))
    print('Min FEXIPRO-SI speedup: ' + str(np.min(all_fexipro_si_speedups)))
    print('Max FEXIPRO-SI speedup: ' + str(np.max(all_fexipro_si_speedups)))

    all_overheads = np.ravel(all_overheads)
    print('Average overhead: ' + str(np.mean(all_overheads)))
    print('Min overhead: ' + str(np.min(all_overheads)))
    print('Max overhead: ' + str(np.max(all_overheads)))

    all_percent_overheads = np.ravel(all_percent_overheads)
    print('Average percent overhead: ' + str(np.mean(all_percent_overheads)))
    print('Min percent overhead: ' + str(np.min(all_percent_overheads)))
    print('Max percent overhead: ' + str(np.max(all_percent_overheads)))

    legend = ax_arr[0].legend(
        loc='upper center',
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=plt.gcf().transFigure,
        ncol=6)
    plt.tight_layout(pad=0.2, w_pad=0.3, h_pad=0.8)
    save_figure('f-u-plot', (legend, ))
    plt.show()


def appetizer_bar_plots(simdex_df,
                        lemp_df,
                        blocked_mm_df,
                        models,
                        model_labels,
                        K=1,
                        bbox_to_anchor=(0, 0, 1, 1),
                        figsize=(15, 10)):

    assert (len(models) == len(model_labels))
    plt.figure(figsize=figsize)

    simdex_rt = simdex_df.sort_values(by='comp_time').groupby(
        ['model', 'K'], as_index=False).first()
    simdex_rt['algo'] = 'Our Index'

    blocked_mm_rt = blocked_mm_df[['model', 'K', 'comp_time']]
    blocked_mm_rt['algo'] = 'Blocked MM'

    lemp_rt = lemp_df[['model', 'K', 'comp_time']]
    lemp_rt['algo'] = 'LEMP'

    query_string = '(' + ('or '.join([('model == "%s"' % model)
                                      for model in models])) + ')'
    query_string += ' and K == %d' % K
    lemp_model_rt = lemp_rt.query(query_string)
    simdex_model_rt = simdex_rt.query(query_string)
    blocked_mm_model_rt = blocked_mm_rt.query(query_string)

    data = pd.concat([lemp_model_rt, blocked_mm_model_rt, simdex_model_rt])
    if len(data) == 0: return
    max_runtime = max(lemp_model_rt['comp_time'].max(),
                      blocked_mm_model_rt['comp_time'].max())
    sns.barplot(
        x='model',
        y='comp_time',
        hue='algo',
        data=data,
        linewidth=1.25,
        edgecolor='black')
    legend = plt.legend(
        loc='upper center',
        bbox_to_anchor=bbox_to_anchor,
        borderaxespad=0.,
        ncol=3)

    start, end = plt.ylim()
    plt.xticks(plt.xticks()[0], model_labels)
    plt.ylim([start, max_runtime * 1.1])
    plt.minorticks_on()
    plt.xlabel('Model')
    plt.ylabel('Time (s)')
    plt.grid(True)
    # plt.title(LABEL_DICT[model], y=y_title)
    sns.despine()

    save_figure('appetizer-bar', (legend, ))
    plt.show()


def rmse_and_reg_plots(blocked_mm_df,
                       lemp_df,
                       rmse_df,
                       model_prefix,
                       regs,
                       simdex_df=None,
                       fexipro_df=None,
                       fexipro_si_df=None,
                       K=1,
                       fname=None,
                       figsize=(28, 6.5),
                       bbox_to_anchor=(0, 0, 1, 1),
                       title=False,
                       title_text=None,
                       y_title=-0.4,
                       annotate=True,
                       xy_text=(15, 150),
                       linestyle='--',
                       markerstyle='x',
                       include_legend=False):
    blocked_mm_rt = blocked_mm_df.query('K == %d' % K)[['model', 'comp_time']]
    lemp_rt = lemp_df.query('K == %d' % K)[['model', 'comp_time']]

    if simdex_df is not None:
        simdex_rt = simdex_df.query('K == %d' % K).sort_values(
            by='comp_time').groupby(
                ['model', 'K'], as_index=False).first()[['model', 'comp_time']]
    if fexipro_df is not None:
        fexipro_rt = fexipro_df.query('K == %d' % K)[['model', 'comp_time']]
    if fexipro_si_df is not None:
        fexipro_si_rt = fexipro_si_df.query(
            'K == %d' % K)[['model', 'comp_time']]

    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, sharex=True, figsize=figsize)
    blocked_mm_data = []
    lemp_data = []
    if simdex_df is not None:
        simdex_data = []
    if fexipro_df is not None:
        fexipro_data = []
    if fexipro_si_df is not None:
        fexipro_si_data = []
    rmse_data = []

    for reg in regs:
        if reg == 'gold-standard':
            model = model_prefix + '-' + reg
            reg = '1'
        else:
            model = model_prefix + '-reg-' + reg

        test_rmse = rmse_df.query('model == "%s"' % model)['test_rmse'].min()
        rmse_data.append(test_rmse)

        blocked_mm_result = blocked_mm_rt.query(
            'model == "%s"' % model)['comp_time'].min()
        blocked_mm_data.append(blocked_mm_result)

        lemp_result = lemp_rt.query('model == "%s"' % model)['comp_time'].min()
        lemp_data.append(lemp_result)

        if simdex_df is not None:
            simdex_result = simdex_rt.query(
                'model == "%s"' % model)['comp_time'].min()
            simdex_data.append(simdex_result)

        if fexipro_df is not None:
            fexipro_result = fexipro_rt.query(
                'model == "%s"' % model)['comp_time'].min()
            fexipro_data.append(fexipro_result)

        if fexipro_si_df is not None:
            fexipro_si_result = fexipro_si_rt.query(
                'model == "%s"' % model)['comp_time'].min()
            fexipro_si_data.append(fexipro_si_result)

    #if fexipro_df is not None:
    #    max_runtime = min(fexipro_data) # Fexipro will be cut off, except for the smallest value
    #else:
    #    max_runtime = max([max(blocked_mm_data), max(lemp_data),])

    min_rmse = min(rmse_data)
    best_reg_index = rmse_data.index(min_rmse)
    regs = [float(x) for x in regs]
    print(regs)
    if regs[0] == 0:
        regs = np.asarray(regs) + 1e-8
    best_reg = regs[best_reg_index]

    ax1.plot(
        regs,
        rmse_data,
        color='black',
        linestyle=linestyle,
        marker=markerstyle)
    ax1.set_xscale('symlog')
    ax1.set_ylabel('Test RMSE', labelpad=5)
    ax1.grid(True)
    ax1.minorticks_on()
    if annotate:
        ax1.annotate(
            'best model',
            xy=(best_reg, min_rmse
                ),  # min_rmse * 1.001 for R2-nomad 25, min_rmse otherwise
            xycoords='data',
            xytext=xy_text,
            textcoords='offset points',
            arrowprops=dict(
                facecolor='black', shrink=0.11
            ),  # 0.03 for R2-nomad 25, 0.06 for R2-nomad 100, 0.11 otherwise,
            horizontalalignment=
            'center',  # right for R2-nomad 25, left for R2-nomad 100, center otherwise
            verticalalignment='right',
            fontsize=20)

    min_rt = min([blocked_mm_data[best_reg_index], lemp_data[best_reg_index]])

    num_legend_entries = 2
    ax2.plot(
        regs,
        blocked_mm_data,
        label='Blocked MM',
        linestyle=linestyle,
        marker=markerstyle)
    if simdex_df is not None:
        num_legend_entries += 1
        ax2.plot(
            regs,
            simdex_data,
            label='Maximus',
            linestyle=linestyle,
            marker=markerstyle)

    ax2.plot(
        regs, lemp_data, label='LEMP', linestyle=linestyle, marker=markerstyle)
    if fexipro_df is not None:
        num_legend_entries += 1
        ax2.plot(
            regs,
            fexipro_data,
            label='FEXIPRO-SIR',
            linestyle=linestyle,
            marker=markerstyle)
    if fexipro_si_df is not None:
        num_legend_entries += 1
        ax2.plot(
            regs,
            fexipro_si_data,
            label='FEXIPRO-SI',
            linestyle=linestyle,
            marker=markerstyle)
    if annotate:
        # Comment out for the non-appetizer plots
        ax2.annotate(
            'best runtime\non best model',
            xy=(best_reg, min_rt),
            xycoords='data',
            xytext=xy_text,
            textcoords='offset points',
            arrowprops=dict(facecolor='black',
                            shrink=0.11),  #0.08 for R2-nomad-25
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=20)
    if include_legend:
        ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    if title:
        if title_text is None:
            title_text = LABEL_DICT[model_prefix]
        if y_title < 0.0:
            ax2.set_title(title_text, y=y_title)
        else:
            fig.suptitle(title_text, x=0.6, y=y_title)

    sns.despine()
    ax2.set_xscale('log')
    ax2.set_xlabel(r'Regularization', labelpad=5)
    ax2.set_ylabel('Time (s)', labelpad=5)
    start, end = ax2.get_ylim()
    #ax2.set_ylim([start, max_runtime * 1.1])
    ax2.grid(True)
    ax2.minorticks_on()

    fig.tight_layout()
    if fname:
        save_figure(fname)
    else:
        save_figure('rmse-reg-' + model_prefix)
    plt.show()


# save legend separately
def rmse_and_reg_legend(add_simdex=False, linestyle='--', markerstyle='x'):
    def flip(items, ncol):
        return itertools.chain(* [items[i::ncol] for i in range(ncol)])

    labels = ['Model Error', 'Blocked MM', 'LEMP', 'FEXIPRO-SIR', 'FEXIPRO-SI']
    ncol = len(labels)
    if add_simdex:

        labels = labels[:2] + ['Maximus'] + labels[2:]
        ncol += 1
        # ncol = int(len(labels) / 2)

        fig = pylab.figure()
        figlegend = pylab.figure(figsize=(22, 1.5))
        ax = fig.add_subplot(111)
        lines = ax.plot(
            range(10),
            pylab.randn(10),
            'black',
            range(10),
            pylab.randn(10),
            range(10),
            pylab.randn(10),
            range(10),
            pylab.randn(10),
            range(10),
            pylab.randn(10),
            range(10),
            pylab.randn(10),
            linestyle=linestyle,
            marker=markerstyle)
        figlegend.legend(
            flip(lines, ncol),
            flip(labels, ncol),
            loc='center',
            ncol=ncol,
            columnspacing=None,
            labelspacing=None)
        figlegend.savefig(FIGURES_DIR + 'rmse-reg-legend.pdf')

    else:
        fig = pylab.figure()
        figlegend = pylab.figure(figsize=(18, 1.5))
        ax = fig.add_subplot(111)
        lines = ax.plot(
            range(10),
            pylab.randn(10),
            'black',
            range(10),
            pylab.randn(10),
            range(10),
            pylab.randn(10),
            range(10),
            pylab.randn(10),
            range(10),
            pylab.randn(10),
            linestyle=linestyle,
            marker=markerstyle)
        figlegend.legend(
            lines,
            labels,
            loc='center',
            ncol=ncol,
            columnspacing=None,
            labelspacing=None)
        figlegend.savefig(FIGURES_DIR + 'appetizer-legend.pdf')


def num_clusters_vs_runtime_single(simdex_df,
                                   model,
                                   nrows=1,
                                   filter_value=4096,
                                   figsize=(28, 6.5),
                                   bbox_to_anchor=(0, 0, 1, 1),
                                   y_title=-0.25):

    num_legend_entries = -1

    table = simdex_df.query('model == "%s" and num_clusters < %d' %
                            (model, filter_value))
    if len(table) == 0: return
    data = table.groupby(['num_clusters', 'K'], as_index=False).aggregate(min)
    num_legend_entries = max(num_legend_entries,
                             len(data['num_clusters'].unique()))
    sns.barplot(
        x='K',
        y='comp_time',
        hue='num_clusters',
        data=data,
        linewidth=1.25,
        edgecolor='black')

    sns.despine()
    plt.title(model, y=y_title)
    plt.xlabel('K')
    plt.ylabel('Time (s)')
    plt.grid(True)
    plt.minorticks_on()

    legend = plt.legend(loc='upper left', bbox_to_anchor=bbox_to_anchor)
    modify_legend(legend, [('$C=', '$')])

    save_figure('n-clusters-vs-runtime-%s' % model, (legend, ))
    plt.show()


def num_clusters_vs_runtime(simdex_df,
                            models,
                            nrows=1,
                            min_value=0,
                            max_value=4096,
                            figsize=(28, 6.5),
                            bbox_to_anchor=(0, 0, 1, 1),
                            y_title=-0.25):

    num_legend_entries = -1
    fig, ax_arr = plt.subplots(
        nrows=nrows, ncols=int(len(models) / nrows), figsize=figsize)

    for i, model in enumerate(models):
        table = simdex_df.query(
            'model == "%s" and num_clusters < %d and num_clusters > %d' %
            (model, max_value, min_value))
        if len(table) == 0: continue
        data = table.groupby(
            ['num_clusters', 'K'], as_index=False).aggregate(min)
        num_legend_entries = max(num_legend_entries,
                                 len(data['num_clusters'].unique()))
        sns.barplot(
            x='K',
            y='comp_time',
            hue='num_clusters',
            data=data,
            linewidth=1.25,
            edgecolor='black',
            ax=ax_arr[i])

        ax_arr[i].legend_.remove()
        sns.despine()
        ax_arr[i].set_title(LABEL_DICT[model], y=y_title)
        ax_arr[i].set_xlabel('K')
        if i == 0:
            ax_arr[i].set_ylabel('Time (s)')
        else:
            ax_arr[i].set_ylabel('')
        ax_arr[i].grid(True)
        ax_arr[i].minorticks_on()

    legend = ax_arr[0].legend(
        loc='upper center',
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=plt.gcf().transFigure,
        ncol=num_legend_entries)
    modify_legend(legend, [('$C=', '$')])

    save_figure('n-clusters-vs-runtime', (legend, ))
    plt.show()


def batch_size_vs_runtime_single(simdex_df,
                                 model,
                                 figsize=(28, 6.5),
                                 bbox_to_anchor=(0, 0, 1, 1),
                                 y_title=-0.25):
    best_rt = simdex_df.sort_values(by='comp_time').groupby(
        ['model', 'K'], as_index=False).first()

    best_rt_model = best_rt.query('model == "%s"' % model)
    if len(best_rt_model) == 0: return
    data = []
    for _, row in best_rt_model.iterrows():
        K, num_clusters = row['K'], row['num_clusters']
        data_for_K = simdex_df.query(
            'model == "%s" and num_clusters == %d and K == %d' %
            (model, num_clusters, K))
        data_for_K = data_for_K.sort_values(by='batch_size')
        data.append(data_for_K)
    data = pd.concat(data)
    sns.barplot(
        x='K',
        y='comp_time',
        hue='batch_size',
        data=data,
        linewidth=1.25,
        edgecolor='black',
        ci=None)

    sns.despine()
    plt.title(LABEL_DICT[model], y=y_title)
    plt.xlabel('K')
    plt.ylabel('Time (s)')
    plt.grid(True)
    plt.minorticks_on()

    legend = plt.legend(
        loc='upper center',
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=plt.gcf().transFigure)
    modify_legend(legend, [('Batch Size = ', '')])

    save_figure('batch-size-vs-runtime', (legend))
    plt.show()


###
# Not the same as num_clusters_vs_runtime: We find the best runtime and the
# associated num_clusters; then for that value of num_clusters, we get all
# the batch_sizes
###
def batch_size_vs_runtime(simdex_df,
                          models,
                          nrows=1,
                          figsize=(28, 6.5),
                          bbox_to_anchor=(0, 0, 1, 1),
                          y_title=-0.25):
    best_rt = simdex_df.sort_values(by='comp_time').groupby(
        ['model', 'K'], as_index=False).first()
    num_legend_entries = len(simdex_df['batch_size'].unique())
    fig, ax_arr = plt.subplots(
        nrows=nrows,
        ncols=int(len(models) / nrows),
        sharey=True,
        figsize=figsize)

    for i, model in enumerate(models):
        best_rt_model = best_rt.query('model == "%s"' % model)
        if len(best_rt_model) == 0: continue
        data = []
        for _, row in best_rt_model.iterrows():
            K, num_clusters = row['K'], row['num_clusters']
            #print(model, K, num_clusters)
            data_for_K = simdex_df.query(
                'model == "%s" and num_clusters == %d and K == %d' %
                (model, num_clusters, K))
            data_for_K = data_for_K.sort_values(by='comp_time').groupby(
                ['K', 'batch_size'], as_index=False).first()
            if len(data_for_K['batch_size']
                   .unique()) != 5:  # [256, 512, 1024, 2048, 4096]
                print('Using 8 clusters!')
                # instead of using the best cluster, use 8 clusters
                data_for_K = simdex_df.query(
                    'model == "%s" and num_clusters == %d and K == %d' %
                    (model, 8, K))
                data_for_K = data_for_K.sort_values(by='comp_time').groupby(
                    ['K', 'batch_size'], as_index=False).first()
            data.append(data_for_K)
        data = pd.concat(data)
        #print(model, data[['K', 'batch_size', 'comp_time']])
        sns.barplot(
            x='K',
            y='comp_time',
            hue='batch_size',
            data=data,
            linewidth=1.25,
            edgecolor='black',
            ci=None,
            ax=ax_arr[i])

        ax_arr[i].legend_.remove()
        sns.despine()
        ax_arr[i].set_title(LABEL_DICT[model], y=y_title)
        ax_arr[i].set_xlabel('K')
        if i == 0:
            ax_arr[i].set_ylabel('Time (s)')
        else:
            ax_arr[i].set_ylabel('')
        ax_arr[i].grid(True)
        ax_arr[i].minorticks_on()

    legend = plt.legend(
        loc='upper center',
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=plt.gcf().transFigure,
        ncol=num_legend_entries)
    modify_legend(legend, [('$B=', '$')])

    save_figure('batch-size-vs-runtime', (legend, ))
    plt.show()


def num_bins_vs_runtime(simdex_df,
                        models,
                        nrows=1,
                        figsize=(28, 6.5),
                        bbox_to_anchor=(0, 0, 1, 1),
                        y_title=-0.25):
    best_rt = simdex_df.sort_values(by='comp_time').groupby(
        ['model', 'K'], as_index=False).first()
    num_legend_entries = len(best_rt['num_bins'].unique())
    fig, ax_arr = plt.subplots(
        nrows=nrows,
        ncols=int(len(models) / nrows),
        sharey=True,
        figsize=figsize)

    for i, model in enumerate(models):
        best_rt_model = best_rt.query('model == "%s"' % model)
        if len(best_rt_model) == 0: continue
        data = []
        for _, row in best_rt_model.iterrows():
            K, num_clusters = row['K'], row['num_clusters']
            data_for_K = simdex_df.query(
                'model == "%s" and num_clusters == %d and K == %d' %
                (model, num_clusters, K))
            data_for_K = data_for_K.sort_values(by='num_bins')
            data.append(data_for_K)
        data = pd.concat(data)
        sns.barplot(
            x='K',
            y='comp_time',
            hue='num_bins',
            data=data,
            linewidth=1.25,
            edgecolor='black',
            ci=None,
            ax=ax_arr[i])

        ax_arr[i].legend_.remove()
        sns.despine()
        ax_arr[i].set_title(LABEL_DICT[model], y=y_title)
        ax_arr[i].set_xlabel('K')
        if i == 0:
            ax_arr[i].set_ylabel('Time (s)')
        else:
            ax_arr[i].set_ylabel('')
        ax_arr[i].grid(True)
        ax_arr[i].minorticks_on()

    legend = plt.legend(
        loc='upper center',
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=plt.gcf().transFigure,
        ncol=num_legend_entries)
    modify_legend(legend, [('$B=', '$')])

    #fig.subplots_adjust(hspace=0)
    save_figure('n-bins-vs-runtime', (legend))
    plt.show()


def point_query_time(models,
                     csv_dir='user-stats/K-1',
                     sample_fraction=0.1,
                     bins=10,
                     nrows=1,
                     figsize=(32, 4.5),
                     y_title=-0.40):
    if nrows == 1:
        fig, ax_arr = plt.subplots(
            nrows=nrows,
            ncols=int(len(models) / nrows),
            sharey=True,
            figsize=figsize)
    else:
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 6)  # hard-coded to 6 columns for now,
        # optimized for the 5-suplot case

        ax1 = plt.subplot(gs[0, 0:2])
        ax2 = plt.subplot(gs[0, 2:4])
        ax3 = plt.subplot(gs[0, 4:6])
        ax4 = plt.subplot(gs[1, 1:3])
        ax5 = plt.subplot(gs[1, 3:5])

        ax_arr = [ax1, ax2, ax3, ax4, ax5]

    for i, model in enumerate(models):
        for filename in glob.iglob('%s/%s_user_stats_*' % (csv_dir, model)):
            df = pd.read_csv(filename)
            x = df['query_time']
            print(np.mean(x), len(x))
            sorted_x = sorted(x.sample(int(len(x) * sample_fraction)))
            N = len(sorted_x)
            y_values = np.arange(N) / float(N)
            ax_arr[i].plot(sorted_x, y_values)
            #sns.distplot(x, kde=False, rug=False, bins=bins, color=color,
            #        hist_kws={'alpha': 0.8,}, ax=ax_arr[i])
            #sns.despine()
            xlim = ax_arr[i].get_xlim()
            ax_arr[i].set_xlim([0, xlim[-1]])
            ax_arr[i].set_title(LABEL_DICT[model], y=y_title)
            ax_arr[i].grid(True)
            ax_arr[i].minorticks_on()
            ax_arr[i].set_ylim([0, 1.05])
            if i == 0:
                ax_arr[i].set_ylabel('CDF')
            else:
                ax_arr[i].set_ylabel('')
            ax_arr[i].set_xlabel('Query time per user (ms)')
            break
    fig.tight_layout()
    if nrows == 2:
        gs.tight_layout(fig)
        save_figure('point-query-time', tight=False)
    else:
        save_figure('point-query-time', tight=True)
    plt.show()


def reg_vs_runtime(simdex_df,
                   blocked_mm_df,
                   model_prefixes,
                   regs,
                   fname,
                   nrows=1,
                   figsize=(28, 6.5),
                   y_title=-0.25):
    simdex_rt = simdex_df.sort_values(by='comp_time').groupby(
        ['model', 'K'], as_index=False).first()[['model', 'K', 'comp_time']]
    blocked_mm_rt = blocked_mm_df[['model', 'K', 'comp_time']]

    fig, ax_arr = plt.subplots(
        nrows=nrows, ncols=int(len(model_prefixes) / nrows), figsize=figsize)
    for i, model_prefix in enumerate(model_prefixes):
        simdex_data = []
        blocked_mm_data = []
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
            simdex_data.append(simdex_result)

            blocked_mm_result = blocked_mm_rt.query('model == "%s"' % model)
            blocked_mm_result['reg'] = 'Blocked MM Only'
            blocked_mm_data.append(blocked_mm_result)

        data = pd.concat(simdex_data + blocked_mm_data)

        if len(data) == 0: continue
        max_runtime = data['comp_time'].max()
        sns.barplot(
            x='K',
            y='comp_time',
            hue='reg',
            data=data,
            linewidth=1.25,
            edgecolor='black',
            ci=None,
            ax=ax_arr[i])
        # add_hatches(bar, num_groups=4)

        if i < len(model_prefixes) - 1:
            ax_arr[i].legend_.remove()
        else:
            legend = ax_arr[i].legend(
                bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            legend_pref_suf = [(r'''Index Only,
$\lambda = ''', '$'), (r'''Index Only,
$\lambda = ''', '$'), (r'''Index Only,
$\lambda = ''', '$'), (r'''Index Only,
$\lambda = ''', '$'), ('', r''',
all $\lambda$''')]
            modify_legend(legend, legend_pref_suf)

        sns.despine()
        ax_arr[i].set_title(LABEL_DICT[model_prefix], y=y_title)
        ax_arr[i].set_xlabel('K')
        if i == 0:
            ax_arr[i].set_ylabel('Time (s)')
        else:
            ax_arr[i].set_ylabel('')
        start, end = ax_arr[i].get_ylim()
        ax_arr[i].set_ylim([start, max_runtime * 1.1])
        ax_arr[i].grid(True)
        ax_arr[i].minorticks_on()

    save_figure('reg-%s' % fname, (legend))
    plt.show()


def decision_rule(simdex_df,
                  blocked_mm_df,
                  models,
                  K,
                  column='theta_uc',
                  num_clusters_to_plot=None,
                  x_line=None,
                  y_line=None,
                  xlabel=None,
                  title=None,
                  fname=None,
                  log=False,
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
        for filename in glob.iglob('%s/%s_user_stats_*' % (csv_dir, model)):
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
            line_style = '--' if blocked_mm_rt.values[0] < simdex_rt.values[0] else '-'
            vals.append((theta_bs, model, line_style))
            #vals.append((theta_bs, '%s, C=%d' % (model, num_clusters),
            #    line_style))
            break
        return vals

    vals = []
    for model in models:
        for val in get_vals(model):
            vals.append(val)
    if fname:
        fname = 'cdf-%s' % fname
    if not xlabel:
        xlabel = column
    plot_cdf(
        vals,
        xlabel,
        x_line=x_line,
        y_line=y_line,
        title=title,
        log=log,
        fname=fname,
        show=True)


def dataset_size_vs_runtime(simdex_df, lemp_df, blocked_mm_df, axis='users'):
    def get_rt_df(df, algo):
        rt_df = df[blocked_mm_df['model'].str.contains(axis)][[
            'model', 'K', 'comp_time'
        ]]
        rt_df['x'] = rt_df['model'].map(lambda x: float(x[-9:-6]))
        rt_df['algo'] = algo
        return rt_df

    simdex_rt_df = get_rt_df(
        simdex_df.sort_values(by='comp_time').groupby(
            ['model', 'K'], as_index=False).first().drop(
                [
                    'num_threads', 'num_latent_factors', 'batch_size',
                    'sample_percentage', 'num_iters'
                ],
                axis=1),
        'SimDex-Index OnlyOnly')
    lemp_rt_df = get_rt_df(lemp_df, 'LEMP')
    blocked_mm_rt_df = get_rt_df(blocked_mm_df, 'SimDex-Blocked MM Only')

    for df in [simdex_rt_df, blocked_mm_rt_df, lemp_rt_df]:
        plt.plot(df['x'], df['comp_time'], label=df['algo'].unique()[0])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.minorticks_on()
    plt.xlabel(axis)
    ylim = plt.ylim()
    plt.ylim([0, ylim[-1]])
    plt.xlim([0.1, 1.0])
    plt.grid(True)
    sns.despine()
    plt.show()


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
