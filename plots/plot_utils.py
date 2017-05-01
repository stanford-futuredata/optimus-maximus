from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
import os
import glob
from itertools import cycle

FIGURES_DIR = 'figures/'
if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)


LABEL_DICT = {
    'lemp-paper-Netflix-noav-10': r'Teflioudi et al., Netflix, $f=10$',
    'nomad-Netflix-10': r'Yun et al., Netflix, $f=10$',
    'nomad-Netflix-10-reg-0.05': r'Yun et al., Netflix, $f=10$',
    'nomad-Netflix-25': r'Yun et al., Netflix, $f=25$',
    'nomad-Netflix-25-reg-0.05': r'Yun et al., Netflix, $f=25$',
    'lemp-paper-Netflix-50': r'Teflioudi et al., Netflix, with avg., $f=50$',
    'lemp-paper-Netflix-noav-50': r'Teflioudi et al., Netflix, $f=50$',
    'nomad-Netflix-50': r'Yun et al., Netflix, $f=50$',
    'nomad-Netflix-50-reg-0.05': r'Yun et al., Netflix, $f=50$',
    'lemp-paper-Netflix-noav-100': r'Teflioudi et al., Netflix, $f=100$',
    'nomad-Netflix-100': r'Yun et al., Netflix, $f=100$',
    'nomad-Netflix-100-gold-standard': r'Yun et al., Netflix, $f=100$',
    'nomad-KDD-10-reg-1': r'Yun et al., KDD, $f=10$',
    'nomad-KDD-25-reg-0.001': r'Yun et al., KDD, $f=25$',
    'nomad-KDD-50-reg-1': r'Yun et al., KDD, $f=50$',
    'lemp-paper-KDD-50': r'Teflioudi et al., KDD, $f=50$',
    'nomad-KDD-100-reg-1': r'Yun et al., KDD, $f=100$',
    'nomad-R2-10': r'R2, $f=10$',
    'nomad-R2-10-reg-0.001': r'R2, $f=10$',
    'nomad-R2-25': r'R2, $f=25$',
    'nomad-R2-25-reg-0.001': r'R2, $f=25$',
    'nomad-R2-50': r'R2, $f=50$',
    'nomad-R2-50-reg-0.001': r'R2, $f=50$',
    'nomad-R2-100': r'R2, $f=100$',
    'nomad-R2-100-reg-0.01': r'R2, $f=100$',
}


HATCHES = ['0', '-', '/', 'x', '.', '|', '+', '//', '\\', 'o', '*']

###########
## UTILS ##
###########
def save_figure(filename, legend=None, tight=True):
    filename = filename.replace('.', '-')
    filename += '.pdf'
    if legend:
        plt.savefig(
            FIGURES_DIR + filename,
            bbox_extra_artists=(legend, ),
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


#sns.set_palette('colorblind', n_colors=7)
def plot_cdf(values_labels_and_lines,
             xlabel,
             x_line=None,
             title=None,
             log=False,
             show=True,
             fname=None):
    #linestyles = cycle(['-', '--', '-.', ':'])
    for (_x_values, label, line_style) in values_labels_and_lines:
        x_values = sorted(_x_values)
        N = len(x_values)
        y_values = np.arange(N) / float(N)
        plt.plot(x_values, y_values, label=label, linestyle=line_style)
    if x_line:
        plt.axvline(x=x_line, color='black', linestyle='-', linewidth=2.5)
    if log:
        plt.xscale('log')
    xlim = plt.xlim()
    plt.xlim([0.0, xlim[-1]])
    yticks = np.arange(0.0, 1.05, 0.15)
    yticks[-1] = 1.0
    plt.yticks(yticks)
    plt.ylim([0.0, 1.0])
    plt.minorticks_on()
    if title:
        plt.title(title)
    legend = plt.legend(bbox_to_anchor=(0.5, -0.17), loc='upper center', borderaxespad=0.)
    replace_legend_labels(legend)
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
def f_u_plots(simdex_df, lemp_df, blocked_mm_df, models, nrows=1):
    if len(models) == 5:
        fig = plt.figure(figsize=(28, 15))
        gs = gridspec.GridSpec(2, 6)

        ax1 = plt.subplot(gs[0, 0:2])
        ax2 = plt.subplot(gs[0, 2:4])
        ax3 = plt.subplot(gs[0, 4:6])
        ax4 = plt.subplot(gs[1, 1:3])
        ax5 = plt.subplot(gs[1, 3:5])

        gs.tight_layout(fig, h_pad=4.25)
        ax_arr = [ax1,ax2,ax3,ax4,ax5]
    else:
        fig, ax_arr = plt.subplots(nrows=nrows, ncols=int(len(models) / nrows), figsize=(28, 6.5))


    both_df = pd.concat([simdex_df[['model', 'K', 'comp_time']], blocked_mm_df[['model', 'K', 'comp_time']]])
    both_rt = both_df.sort_values(by='comp_time').groupby( ['model', 'K'], as_index=False).first()
    both_rt['algo'] = 'SimDex'

    simdex_rt = simdex_df.sort_values(by='comp_time').groupby( ['model', 'K'], as_index=False).first()
    simdex_rt['algo'] = 'Index Only'

    blocked_mm_rt = blocked_mm_df[['model', 'K', 'comp_time']]
    blocked_mm_rt['algo'] = 'Blocked MM Only'

    lemp_rt = lemp_df[['model', 'K', 'comp_time']]
    lemp_rt['algo'] = 'LEMP'

    for i, model in enumerate(models):
        both_model_rt = both_rt.query('model == "%s"' %
                model).sort_values(by='K')
        lemp_model_rt = lemp_rt.query('model == "%s"' % model).sort_values(by='K')
        simdex_model_rt = simdex_rt.query('model == "%s"' % model).sort_values(by='K')
        blocked_mm_model_rt = blocked_mm_rt.query('model == "%s"' % model).sort_values(by='K')

        speed_ups = lemp_model_rt['comp_time'] / both_model_rt['comp_time']
        print model
        print speed_ups
        data = pd.concat([both_model_rt, lemp_model_rt, blocked_mm_model_rt,
            simdex_model_rt])
        if len(data) == 0: continue
        max_runtime = data['comp_time'].max()
        sns.barplot(
            x='K',
            y='comp_time',
            hue='algo',
            data=data,
            linewidth=1.25,
            edgecolor='black',
            ax=ax_arr[i])
        #add_hatches(bar, num_groups=4)

        if i < len(models)-1:
            ax_arr[i].legend_.remove()
        else:
            legend = ax_arr[i].legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0.)

        start, end = ax_arr[i].get_ylim()
        ax_arr[i].set_ylim([start, max_runtime * 1.1])
        ax_arr[i].minorticks_on()
        ax_arr[i].set_xlabel('K')
        if i == 0:
            ax_arr[i].set_ylabel('Time (s)')
        else:
            ax_arr[i].set_ylabel('')
        ax_arr[i].grid(True)
        ax_arr[i].set_title(LABEL_DICT[model], y=-0.22)
        sns.despine()
    save_figure('f-u-plot', legend)
    plt.show()


def num_clusters_vs_runtime(simdex_df, models, nrows=1, filter_value=4096):
    fig, ax_arr = plt.subplots(nrows=nrows, ncols=int(len(models) / nrows), figsize=(28, 6.5))
    for i, model in enumerate(models):
        table = simdex_df.query('model == "%s" and num_clusters < %d' % (model,
            filter_value))
        if len(table) == 0: continue
        data = table.groupby(
            ['num_clusters', 'K'], as_index=False).aggregate(min)
        sns.barplot(
            x='K',
            y='comp_time',
            hue='num_clusters',
            data=data,
            linewidth=1.25,
            edgecolor='black',
            ax=ax_arr[i])
        sns.barplot(
            x='K',
            y='cluster_time',
            hue='num_clusters',
            data=data,
            linewidth=1.25,
            edgecolor='black',
            color='#ffffff',
            hatch='x',
            ax=ax_arr[i])
        if i < len(models)-1:
            ax_arr[i].legend_.remove()
        else:
            handles, labels = ax_arr[i].get_legend_handles_labels()
            num_legend_entries = int(len(handles) / 2) + 1
            labels[num_legend_entries - 1] = ''
            legend = ax_arr[i].legend(handles[:num_legend_entries], labels[:num_legend_entries],
                               bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            modify_legend(legend, [('$C=', '$') for j in range(num_legend_entries -
                1)] + [(r'$K$-means', '')])

        sns.despine()
        ax_arr[i].set_title(LABEL_DICT[model], y=-0.25)
        ax_arr[i].set_xlabel('K')
        if i == 0:
            ax_arr[i].set_ylabel('Time (s)')
        else:
            ax_arr[i].set_ylabel('')
        ax_arr[i].grid(True)
        ax_arr[i].minorticks_on()

    save_figure('n-clusters-vs-runtime', legend)
    plt.show()


def num_bins_vs_runtime(simdex_df, models, nrows=1):
    best_rt = simdex_df.sort_values(by='comp_time').groupby(
        ['model', 'K'], as_index=False).first()

    fig, ax_arr = plt.subplots(nrows=nrows, ncols=int(len(models) / nrows), figsize=(28, 6.5))
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

        if i < len(models)-1:
            ax_arr[i].legend_.remove()
        else:
            legend = ax_arr[i].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            modify_legend(legend, [('$B=', '$')])
        sns.despine()
        ax_arr[i].set_title(LABEL_DICT[model], y=-0.25)
        ax_arr[i].set_xlabel('K')
        if i == 0:
            ax_arr[i].set_ylabel('Time (s)')
        else:
            ax_arr[i].set_ylabel('')
        ax_arr[i].grid(True)
        ax_arr[i].minorticks_on()

    save_figure('n-bins-vs-runtime', legend)
    plt.show()


def query_time_histogram(models, csv_dir='user-stats/K-1', sample_size=10000, bins=10, nrows=1):
    color = sns.color_palette()[1]
    if len(models) == 5:
        fig = plt.figure(figsize=(28, 15))
        gs = gridspec.GridSpec(2, 6)

        ax1 = plt.subplot(gs[0, 0:2])
        ax2 = plt.subplot(gs[0, 2:4])
        ax3 = plt.subplot(gs[0, 4:6])
        ax4 = plt.subplot(gs[1, 1:3])
        ax5 = plt.subplot(gs[1, 3:5])

        ax_arr = [ax1,ax2,ax3,ax4,ax5]
    else:
        fig, ax_arr = plt.subplots(nrows=nrows, ncols=int(len(models) / nrows), figsize=(28, 6.5))

    for i, model in enumerate(models):
        for filename in glob.iglob('%s/%s_user_stats_*' % (csv_dir, model)):
            df = pd.read_csv(filename)
            x = df.query('cluster_id > 0')['query_time'].sample(sample_size)
            sns.distplot(x, kde=False, rug=False, bins=bins, color=color,
                    hist_kws={'alpha': 0.8,}, ax=ax_arr[i])
            sns.despine()
            xlim = ax_arr[i].get_xlim()
            if xlim[0] < 0:
                ax_arr[i].set_xlim([0, xlim[-1]])
            ax_arr[i].set_title(LABEL_DICT[model], y=-0.25)
            ax_arr[i].grid(True)
            ax_arr[i].minorticks_on()
            ax_arr[i].set_yscale('log')
            if i == 0:
                ax_arr[i].set_ylabel('Frequency')
            ax_arr[i].set_xlabel('Query time per user (ms)')
            break
    gs.tight_layout(fig, h_pad=4.75)
    save_figure('query-time', tight=False)
    plt.show()


def reg_vs_runtime(simdex_df, blocked_mm_df, model_prefixes, regs, fname, nrows=1):
    simdex_rt = simdex_df.sort_values(by='comp_time').groupby(
        ['model', 'K'], as_index=False).first()[['model', 'K', 'comp_time']]
    blocked_mm_rt = blocked_mm_df[['model', 'K', 'comp_time']]

    fig, ax_arr = plt.subplots(nrows=nrows, ncols=int(len(model_prefixes) / nrows), figsize=(28, 6.5))
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
        #add_hatches(bar, num_groups=4)

        if i < len(model_prefixes)-1:
            ax_arr[i].legend_.remove()
        else:
            legend = ax_arr[i].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            legend_pref_suf = [
(r'''Index Only,
$\lambda = ''', '$'),
(r'''Index Only,
$\lambda = ''', '$'),
(r'''Index Only,
$\lambda = ''', '$'),
(r'''Index Only,
$\lambda = ''', '$'),
('',
r''',
all $\lambda$''')
]
            modify_legend(legend, legend_pref_suf)

        sns.despine()
        ax_arr[i].set_title(LABEL_DICT[model_prefix], y=-0.25)
        ax_arr[i].set_xlabel('K')
        if i == 0:
            ax_arr[i].set_ylabel('Time (s)')
        else:
            ax_arr[i].set_ylabel('')
        start, end = ax_arr[i].get_ylim()
        ax_arr[i].set_ylim([start, max_runtime * 1.1])
        ax_arr[i].grid(True)
        ax_arr[i].minorticks_on()

    save_figure('reg-%s' % fname, legend)
    plt.show()


def cdf(simdex_df,
        blocked_mm_df,
        models,
        K,
        column='theta_uc',
        num_clusters_to_plot=None,
        x_line=None,
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
    if title:
        title = title + ' ' + csv_dir
    if not xlabel:
        xlabel = column
    plot_cdf(vals, xlabel, x_line=x_line, title=title, log=log, fname=fname, show=True)


def dataset_size_vs_runtime(simdex_df, lemp_df,
        blocked_mm_df, axis='users'):
    def get_rt_df(df, algo):
        rt_df = df[blocked_mm_df['model'].str.contains(axis)][['model', 'K', 'comp_time']]
        rt_df['x'] = rt_df['model'].map(lambda x: float(x[-9:-6]))
        rt_df['algo'] = algo
        return rt_df

    simdex_rt_df = get_rt_df(simdex_df.sort_values(by='comp_time').groupby(
            ['model', 'K'], as_index=False).first().drop(['num_threads','num_latent_factors',
                                                          'batch_size','sample_percentage','num_iters'],
                                                          axis=1), 'SimDex-Index OnlyOnly')
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

#def num_clusters_vs_runtime(simdex_df, models, filter_value=4096):
#    for model in models:
#        table = simdex_df.query('model == "%s" and num_clusters < %d' % (model,
#            filter_value))
#        if len(table) == 0: continue
#        data = table.groupby(
#            ['num_clusters', 'K'], as_index=False).aggregate(min)
#        ax = sns.barplot(
#            x='K',
#            y='comp_time',
#            hue='num_clusters',
#            data=data,
#            linewidth=1.25,
#            edgecolor='black')
#        sns.barplot(
#            x='K',
#            y='cluster_time',
#            hue='num_clusters',
#            data=data,
#            linewidth=1.25,
#            edgecolor='black',
#            color='#ffffff',
#            hatch='x')
#        handles, labels = ax.get_legend_handles_labels()
#        num_legend_entries = int(len(handles) / 2) + 1
#        labels[num_legend_entries - 1] = ''
#        legend = ax.legend(handles[:num_legend_entries], labels[:num_legend_entries], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#        modify_legend(legend, [('$C=', '$') for i in range(num_legend_entries -
#            1)] + [(r'$K$-means', '')])
#
#        plt.xlabel('K')
#        plt.ylabel('Time (s)')
#        plt.minorticks_on()
#        plt.grid(True)
#        sns.despine()
#        save_figure('n-clusters-vs-runtime-%s' % model, legend)
#        plt.show()
#
#

#def num_bins_vs_runtime(simdex_df, models):
#    best_rt = simdex_df.sort_values(by='comp_time').groupby(
#        ['model', 'K'], as_index=False).first()
#    for model in models:
#        best_rt_model = best_rt.query('model == "%s"' % model)
#        if len(best_rt_model) == 0: continue
#        data = []
#        for _, row in best_rt_model.iterrows():
#            K, num_clusters = row['K'], row['num_clusters']
#            data_for_K = simdex_df.query(
#                'model == "%s" and num_clusters == %d and K == %d' %
#                (model, num_clusters, K))
#            data_for_K = data_for_K.sort_values(by='num_bins')
#            data.append(data_for_K)
#        data = pd.concat(data)
#
#        sns.barplot(
#            x='K',
#            y='comp_time',
#            hue='num_bins',
#            data=data,
#            linewidth=1.25,
#            edgecolor='black',
#            ci=None)
#
#        legend = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#        modify_legend(legend, [('$B=', '$')])
#        start, end = plt.ylim()
#        plt.xlabel('K')
#        plt.ylabel('Time (s)')
#        plt.minorticks_on()
#        plt.grid(True)
#        sns.despine()
#        save_figure('n-bins-vs-runtime-%s' % model, legend)
#        plt.show()

#def reg_vs_runtime(simdex_df, blocked_mm_df, model_prefixes, regs):
#    simdex_rt = simdex_df.sort_values(by='comp_time').groupby(
#        ['model', 'K'], as_index=False).first()[['model', 'K', 'comp_time']]
#    blocked_mm_rt = blocked_mm_df[['model', 'K', 'comp_time']]
#
#    for model_prefix in model_prefixes:
#        simdex_data = []
#        blocked_mm_data = []
#        for reg in regs:
#            if reg == 'gold-standard':
#                model = model_prefix + '-' + reg
#                if 'Netflix' in model_prefix:
#                    reg = '0.05'
#                else:
#                    reg = '1'
#            else:
#                model = model_prefix + '-reg-' + reg
#
#            simdex_result = simdex_rt.query('model == "%s"' % model)
#            simdex_result['reg'] = reg
#            simdex_data.append(simdex_result)
#
#            blocked_mm_result = blocked_mm_rt.query('model == "%s"' % model)
#            blocked_mm_result['reg'] = 'SimDex-Blocked MM Only'
#            blocked_mm_data.append(blocked_mm_result)
#
#        data = pd.concat(simdex_data + blocked_mm_data)
#
#        if len(data) == 0: continue
#        max_runtime = data['comp_time'].max()
#        sns.barplot(
#            x='K',
#            y='comp_time',
#            hue='reg',
#            data=data,
#            linewidth=1.25,
#            edgecolor='black',
#            ci=None)
#        #add_hatches(bar, num_groups=4)
#
#        legend = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#        legend_pref_suf = [(r'SimDex-Index Only, $\lambda = ', '$'), (r'SimDex-Index Only, $\lambda = ', '$'),
#                (r'SimDex-Index Only, $\lambda = ', '$'), (r'SimDex-Index Only, $\lambda = ', '$'),
#                ('', r', all $\lambda$')]
#        modify_legend(legend, legend_pref_suf)
#        start, end = plt.ylim()
#        plt.ylim([start, max_runtime * 1.1])
#        plt.minorticks_on()
#        plt.xlabel('K')
#        plt.ylabel('Time (s)')
#        plt.grid(True)
#        sns.despine()
#        save_figure('reg-%s' % model_prefix, legend)
#        plt.show()
#
