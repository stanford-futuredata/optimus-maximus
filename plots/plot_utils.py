
from __future__ import division
from plots_config import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("ticks")
sns.set_palette("RdBu", n_colors=7)

LABEL_DICT = {
    'kdd': 'Yahoo KDD',
    'netflix': 'Netflix',
    'r2': 'Yahoo R2',
    'blocked_naive': 'Blocked',
    'lemp': 'LEMP',
    'naive': 'LEMP Naive',
    'simdex': 'SimDex',
}

F_ = [10, 25, 50]
DATASETS = ['kdd', 'netflix', 'r2']

###########
## UTILS ##
###########
def save_figure(filename, legend=None):
    if legend:
        plt.savefig(filename, bbox_extra_artists=(legend,), bbox_inches='tight')
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

def plot_cdf(values_and_labels, xlabel, title=None, xlim=None, ylim=None, log=False, show=True):
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
    return df.query('K == 1')[['Cluster', 'ThetaUC']].groupby('Cluster', as_index=False).aggregate(max)['ThetaUC']

###########
## PLOTS ##
###########
def f_u_plots(df):
    for dataset in DATASETS:
        for F in F_:
            table = df.query('dataset_name == "%s" and n_features == %d' % (dataset, F))
            data = table.groupby(['algorithm', 'k'], as_index=False).aggregate(min)
            max_runtime = data['total_compute_time'].max()
            _ = sns.barplot(x="k", y="total_compute_time", hue="algorithm", data=data)
            legend = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            fix_legend(legend)
            plt.yscale('log')
            start, end = plt.ylim()
            plt.ylim([start, max_runtime*1.2])
            plt.xlabel('K')
            plt.ylabel('Time (s)')
            sns.despine()
            save_figure('f-u-plot-%s-%d.pdf' % (dataset, F), legend)
            plt.show()


def throughput_vs_n_cluster(df):
    for dataset in DATASETS:
        for F in F_:
            table = df.query('dataset_name == "%s" and n_bins == 1.0 and n_features == %d and algorithm == "simdex"' % (dataset, F))
            data = table.groupby(['n_cluster', 'k'], as_index=False).aggregate(min).query('n_cluster <= 4096')
            _ = sns.barplot(x="k", y="total_compute_time", hue="n_cluster", data=data)
            legend = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            fix_legend(legend)
            plt.yscale('log')
            start, end = plt.ylim()
            plt.xlabel('K')
            plt.ylabel('Time (s)')
            sns.despine()
            save_figure('throughput-vs-n-cluster-%s-%d.pdf' % (dataset, F), legend)
            plt.show()

def items_visited_cdf(df, filename):
    items_visited = df[['K', 'ItemsVisited']].groupby(['K'], as_index=False).aggregate(lambda x: list(x))
    x_values = items_visited['ItemsVisited']
    labels = ['K=%d' % val for val in items_visited['K']]
    vals = zip(x_values, labels)
    sns.set_palette("colorblind", n_colors=7)
    plot_cdf(vals, '# items visited', title, log=True)
    save_figure(filename)
    sns.despine()

def theta_uc_items_visited_scatter(df, filename):
    sns.lmplot('ThetaUC', 'ItemsVisited',
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
    plot_cdf(vals, 'ThetaB','Netflix, C=1024', log=False)
