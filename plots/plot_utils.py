from __future__ import division
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import glob
from itertools import cycle

NETFLIX_MODELS = [
    'lemp-paper-Netflix-noav-10',
    'lemp-paper-Netflix-50',
    'lemp-paper-Netflix-noav-50',
    'lemp-paper-Netflix-noav-100',

    #'sigmod-deadline-Netflix-50',
    #'pb-new-Netflix-10',
    #'sigmod-deadline-Netflix-10',
    #'pb-new-Netflix-25',
    #'sigmod-deadline-Netflix-25',
]

KDD_MODELS = [
    'lemp-paper-KDD-50',
    'nomad-KDD-100-gold-standard',
    'nomad-KDD-100-reg-0.001',
    'nomad-KDD-100-reg-0.01',
    'nomad-KDD-100-reg-0.1',
    'nomad-KDD-10-reg-0.001',
    'nomad-KDD-10-reg-0.01',
    'nomad-KDD-10-reg-0.1',
    'nomad-KDD-10-reg-1',
    'nomad-KDD-25-reg-0.001',
    'nomad-KDD-25-reg-0.01',
    'nomad-KDD-25-reg-0.1',
    'nomad-KDD-25-reg-1',
    'nomad-KDD-50-reg-0.1',
    'nomad-KDD-50-reg-1',

    #'pb-new-kdd-10',
    #'sigmod-deadline-kdd-10',
    #'pb-new-kdd-25',
    #'sigmod-deadline-kdd-25',
    #'pb-new-kdd-50',
    #'sigmod-deadline-kdd-50',
]

R2_MODELS = [
    'nomad-R2-10-reg-0.001',
    'nomad-R2-10-reg-0.01',
    'nomad-R2-10-reg-0.1',
    'nomad-R2-10-reg-1',
    'nomad-R2-25-reg-0.001',
    'nomad-R2-25-reg-0.01',
    'nomad-R2-25-reg-0.1',
    'nomad-R2-25-reg-1',
    'nomad-R2-50-reg-0.001',
    'nomad-R2-50-reg-0.01',
    'nomad-R2-50-reg-0.1',
    'nomad-R2-50-reg-1',
    'nomad-R2-100-reg-0.001',
    'nomad-R2-100-reg-0.01',
    'nomad-R2-100-reg-0.1',
    'nomad-R2-100-reg-1',
]

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


def plot_cdf(values_labels_and_colors,
             xlabel,
             title=None,
             xlim=None,
             ylim=None,
             log=False,
             show=True,
             fname=None):
    linestyles = cycle(['-', '--', '-.', ':'])
    for (_x_values, label, color) in values_labels_and_colors:
        x_values = sorted(_x_values)
        N = len(x_values)
        y_values = np.arange(N) / float(N)
        plt.plot(
            x_values,
            y_values,
            label=label,
            color=color,
            linestyle=next(linestyles))
    if log:
        plt.xscale('log')
    if xlim:
        plt.xlim(xlim)
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


def runtime_vs_num_clusters(simdex_df, models):
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


def runtime_vs_num_bins(simdex_df, models):
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


def theta_uc_cdf(simdex_df, blocked_mm_df, models=NETFLIX_MODELS):
    COLORS = ['red', 'orange', 'green', 'blue', 'black']
    best_simdex_rts = simdex_df.sort_values(by='comp_time').groupby(
        ['model', 'K', 'num_clusters'], as_index=False).first()

    def _theta_uc_cdf(model):
        vals = []
        for filename in glob.iglob('theta-ucs/%s_*' % model):
            df = pd.read_csv(filename)
            num_clusters = df['cluster_id'].max() + 1
            theta_ucs = df['theta_uc'].dropna()
            simdex_faster = []
            for K in [1, 5, 10, 50]:
                simdex_rt = best_simdex_rts.query(
                    'model == "%s" and K == %d and num_clusters == %d' %
                    (model, K, num_clusters))['comp_time']
                blocked_mm_rt = blocked_mm_df.query(
                    'model == "%s" and K == %d' % (model, K))['comp_time']
                simdex_faster.append(simdex_rt.values[0] <= blocked_mm_rt.values[0])
            color = COLORS[np.count_nonzero(simdex_faster)]
            vals.append((theta_ucs, 'C=%d' % num_clusters, color))
        plot_cdf(
            vals, 'Theta_UC', model, log=False, fname='theta-ucs-%s' % model)

    for model in models:
        _theta_uc_cdf(model)


def items_visited_cdf(df, filename):
    items_visited = df[['K', 'ItemsVisited']].groupby(
        ['K'], as_index=False).aggregate(lambda x: list(x))
    x_values = items_visited['ItemsVisited']
    labels = ['K=%d' % val for val in items_visited['K']]
    vals = zip(x_values, labels)
    sns.set_palette('colorblind', n_colors=7)
    plot_cdf(vals, '# items visited', title, log=True)
    save_figure(filename)
    sns.despine()


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
