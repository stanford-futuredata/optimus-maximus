from __future__ import division
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

MODELS = [
    'lemp-paper-Netflix-noav-10',
    'pb-new-Netflix-10',
    'sigmod-deadline-Netflix-10',
    'pb-new-Netflix-25',
    'sigmod-deadline-Netflix-25',
    'lemp-paper-Netflix-50',
    'pb-new-Netflix-50',
    'sigmod-deadline-Netflix-50',
    'lemp-paper-Netflix-noav-50',
    'lemp-paper-Netflix-noav-100',
]


###########
## UTILS ##
###########
def save_figure(filename, legend=None):
    if legend:
        plt.savefig(
            filename, bbox_extra_artists=(legend, ), bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.savefig(filename)


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


def plot_cdf(values_and_labels,
             xlabel,
             title=None,
             xlim=None,
             ylim=None,
             log=False,
             show=True):
    for (_x_values, label) in values_and_labels:
        x_values = sorted(_x_values)
        N = len(x_values)
        y_values = np.arange(N) / float(N)
        plt.plot(x_values, y_values, label=label)
    if log:
        plt.xscale('log')
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    if title:
        plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel(xlabel)
    plt.ylabel('CDF')
    plt.grid(True)


def get_theta_b(df):
    return df.query('K == 1')[['Cluster', 'ThetaUC']].groupby(
        'Cluster', as_index=False).aggregate(max)['ThetaUC']


###########
## PLOTS ##
###########
def f_u_plots(simdex_df, lemp_df, blocked_mm_df):
    simdex_rt = simdex_df.sort_values(by='comp_time').groupby(
        ['model', 'K'], as_index=False).first()[['model', 'K', 'comp_time']]
    simdex_rt['algo'] = 'SimDex'

    lemp_rt = lemp_df[['model', 'K', 'comp_time']]
    lemp_rt['algo'] = 'LEMP'

    blocked_mm_rt = blocked_mm_df[['model', 'K', 'comp_time']]
    blocked_mm_rt['algo'] = 'Blocked Matrix Multiply'

    total_rt = pd.concat([simdex_rt, lemp_rt, blocked_mm_rt])

    for model in MODELS:
        data = total_rt.query('model == "%s"' % model)
        max_runtime = data['comp_time'].max()
        sns.barplot(
            x='K',
            y='comp_time',
            hue='algo',
            data=data,
            linewidth=1.25,
            edgecolor='black')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        start, end = plt.ylim()
        plt.title(model)
        plt.ylim([start, max_runtime * 1.1])
        plt.minorticks_on()
        plt.xlabel('K')
        plt.ylabel('Time (s)')
        sns.despine()
        # save_figure('f-u-plot-%s.pdf' % model, legend)
        plt.show()


def throughput_vs_num_clusters(simdex_df):
    for model in MODELS:
        table = simdex_df.query('model == "%s"' % model)
        data = table.groupby(
            ['num_clusters', 'K'], as_index=False).aggregate(min)
        top = sns.barplot(
            x='K',
            y='comp_time',
            hue='num_clusters',
            data=data,
            linewidth=1.25,
            edgecolor='black')
        bottom = sns.barplot(
            x='K',
            y='cluster_time',
            hue='num_clusters',
            data=data,
            linewidth=1.25,
            edgecolor='black',
            color='#ffffff',
            hatch='x')

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        start, end = plt.ylim()
        plt.xlabel('K')
        plt.ylabel('Time (s)')
        plt.minorticks_on()
        sns.despine()
        plt.title(model)
        # save_figure('throughput-vs-n-cluster-%s-%d.pdf' % (dataset, F), legend)
        plt.show()


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


def theta_b_cdf(df, title):
    theta_b_f_10 = get_theta_b(df2)
    theta_b_f_25 = get_theta_b(df3)
    vals = [(theta_b_f_10, 'F=10'), (theta_b_f_25, 'F=25')]
    plot_cdf(vals, 'ThetaB', 'Netflix, C=1024', log=False)
