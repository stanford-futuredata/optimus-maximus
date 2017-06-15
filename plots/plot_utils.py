from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
import pylab
import os
import glob

FIGURES_DIR = 'figures/'
if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)


LABEL_DICT = {
    'fexipro-paper-Netflix-50': r'''Netflix-libPMF, $f=50$''',
    'fexipro-paper-KDD-50': r'''KDD-libPMF, $f=50$''',
    'lemp-paper-Netflix-noav-10': r'''Netflix-DSGD, $f=10$''',
    'nomad-Netflix-10': r'Netflix-DSGD, $f=10$',
    'nomad-Netflix-10-reg-0.05': r'Netflix-NOMAD, $f=10$',
    'nomad-Netflix-25': r'Netflix-NOMAD, $f=25$',
    'nomad-Netflix-25-reg-0.05': r'Netflix-NOMAD, $f=25$',
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
    'nomad-R2-50-reg-0.001': r'R2-NOMAD, $f=50$',
    'nomad-R2-100': r'R2-NOMAD, $f=100$',
    'nomad-R2-100-reg-0.01': r'R2-NOMAD, $f=100$',
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
    legend = plt.legend(bbox_to_anchor=(0.5, -0.19), loc='upper center',
            borderaxespad=0., frameon=True, edgecolor='black', framealpha=0.6)
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
    people = ('Netflix w/o\nWork Sharing', 'Netflix with\nWork Sharing', 'R2 w/o\nWork Sharing', 'R2 with\nWork Sharing', )
    y_pos = np.array([0, 1.5, 3, 4.5])

    H = 0.25

    y_pos_mid = [1.5*H + y for y in y_pos]

    ax.barh(y_pos, [y0[0], y1[0],  y2[0], y3[0]],linewidth=0.50, edgecolor='black', height=H, align='center', label='Clustering')
    y_pos = [H +y for y in y_pos]

    ax.barh(y_pos, [y0[1], y1[1], y2[1], y3[1]], linewidth=0.50,edgecolor='black', height=H, align='center', label='Index Construction')
    y_pos = [H +y for y in y_pos]

    ax.barh(y_pos, [y0[2], y1[2], y2[2], y3[2]], linewidth=0.50,edgecolor='black', height=H, align='center', label='Cost Estimation')
    y_pos = [H +y for y in y_pos]

    ax.barh(y_pos, [y0[3], y1[3], y2[3], y3[3]], linewidth=0.50,edgecolor='black', height=H, align='center', label='Index Traversal')

    handles, labels = ax.get_legend_handles_labels()
    handles.reverse()
    labels.reverse()
    lgd = ax.legend(handles[::-1], labels[::-1], loc='upper center',
                    bbox_to_anchor=(0,0,1,1.3), ncol=2, frameon=False, columnspacing=None,
                    labelspacing=None)

    ax.set_yticks(y_pos_mid)
    ax.set_yticklabels(people)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Time (seconds, log scale)')
    ax.set_xscale('log')
    ax.grid(True)

    sns.despine()
    save_figure('factor-analysis', (lgd,))

def f_u_plot_single(simdex_df, lemp_df, blocked_mm_df, fexipro_df,
        fexipro_si_df, sampling_df, model, num_clusters=8, batch_size=4096, y_title=-0.35):

    simdex_df = simdex_df.query('num_clusters == %d and batch_size == %d' %
            (num_clusters, batch_size))

    both_df = pd.concat([simdex_df[['model', 'K', 'comp_time']],
        blocked_mm_df[['model', 'K', 'comp_time']].query('model != "nomad-Netflix-25-reg-0.05"')])
    both_rt = both_df.sort_values(by='comp_time').groupby(['model', 'K'], as_index=False).first()
    both_rt['algo'] = 'SimDex'

    simdex_rt = simdex_df.sort_values(by='comp_time').groupby( ['model', 'K'], as_index=False).first()
    simdex_rt['algo'] = 'SimDex-Index Only'

    blocked_mm_rt = blocked_mm_df[['model', 'K', 'comp_time']]
    blocked_mm_rt['algo'] = 'Blocked MM Only'

    lemp_rt = lemp_df[['model', 'K', 'comp_time']]
    lemp_rt['algo'] = 'LEMP'

    fexipro_rt = fexipro_df[['model', 'K', 'comp_time']]
    fexipro_rt['algo'] = 'FEXIPRO'

    fexipro_si_rt = fexipro_si_df[['model', 'K', 'comp_time']]
    fexipro_si_rt['algo'] = 'FEXIPRO-SI'

    both_model_rt = both_rt.query('model == "%s"' %
            model).sort_values(by='K')
    lemp_model_rt = lemp_rt.query('model == "%s"' % model).sort_values(by='K')
    fexipro_model_rt = fexipro_rt.query('model == "%s"' % model).sort_values(by='K')
    fexipro_si_model_rt = fexipro_si_rt.query('model == "%s"' % model).sort_values(by='K')
    simdex_model_rt = simdex_rt.query('model == "%s"' % model).sort_values(by='K')
    blocked_mm_model_rt = blocked_mm_rt.query('model == "%s"' % model).sort_values(by='K')
    sampling_model_rt = sampling_df.query('model == "%s"' % model).sort_values(by='K')

    if len(sampling_model_rt) > 0:
        both_model_rt['comp_time'] += sampling_model_rt['comp_time'].values

    data = pd.concat([both_model_rt, blocked_mm_model_rt, simdex_model_rt,
        lemp_model_rt, fexipro_model_rt, fexipro_si_model_rt])
    if len(data) == 0: return
    max_runtime = sorted([
        lemp_model_rt['comp_time'].max(),
        simdex_model_rt['comp_time'].max(),
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
    if not np.isnan(max_runtime):
        plt.ylim([start, max_runtime * 1.1])
    plt.minorticks_on()
    plt.ylabel('Time (s)')
    plt.xlabel('K')

    plt.grid(True)
    plt.title(LABEL_DICT[model], y=y_title)
    sns.despine()

    legend = plt.legend(loc='2', bbox_to_anchor=(1, 1.05))
    save_figure('f-u-plot-single', (legend,))
    plt.show()


def f_u_plots(simdex_df, lemp_df, blocked_mm_df, fexipro_df, fexipro_si_df, sampling_df, models, nrows=4,
        num_clusters=8, batch_size=4096, figsize=(28, 28), y_title=1.05, bbox_to_anchor=(0,0,1,1)):
    legend = None # save for later
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
    ncols = int(len(models) / nrows)
    fig, ax_arr = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    ax_arr = np.ravel(ax_arr)

    simdex_df = simdex_df.query('num_clusters == %d and batch_size == %d' %
            (num_clusters, batch_size))

    both_df = pd.concat([simdex_df[['model', 'K', 'comp_time']],
        blocked_mm_df[['model', 'K', 'comp_time']].query('model != "nomad-Netflix-25-reg-0.05"')])
    both_rt = both_df.sort_values(by='comp_time').groupby(['model', 'K'], as_index=False).first()
    both_rt['algo'] = 'SimDex'

    simdex_rt = simdex_df.sort_values(by='comp_time').groupby( ['model', 'K'], as_index=False).first()
    simdex_rt['algo'] = 'SimDex-Index Only'

    blocked_mm_rt = blocked_mm_df[['model', 'K', 'comp_time']]
    blocked_mm_rt['algo'] = 'Blocked MM Only'

    lemp_rt = lemp_df[['model', 'K', 'comp_time']]
    lemp_rt['algo'] = 'LEMP'

    fexipro_rt = fexipro_df[['model', 'K', 'comp_time']]
    fexipro_rt['algo'] = 'FEXIPRO'

    fexipro_si_rt = fexipro_si_df[['model', 'K', 'comp_time']]
    fexipro_si_rt['algo'] = 'FEXIPRO-SI'

    all_speedups = []
    all_percent_overheads = []
    all_overheads = []

    for i, model in enumerate(models):
        both_model_rt = both_rt.query('model == "%s"' %
                model).sort_values(by='K')
        lemp_model_rt = lemp_rt.query('model == "%s"' % model).sort_values(by='K')
        fexipro_model_rt = fexipro_rt.query('model == "%s"' % model).sort_values(by='K')
        fexipro_si_model_rt = fexipro_si_rt.query('model == "%s"' % model).sort_values(by='K')
        simdex_model_rt = simdex_rt.query('model == "%s"' % model).sort_values(by='K')
        blocked_mm_model_rt = blocked_mm_rt.query('model == "%s"' % model).sort_values(by='K')
        sampling_model_rt = sampling_df.query('model == "%s"' % model).sort_values(by='K')

        # add sampling overhead
        overheads = simdex_model_rt['cluster_time'].values + \
            simdex_model_rt['index_time'].values + \
            sampling_model_rt['comp_time'].values
        all_overheads.append(overheads)
        percent_overheads = overheads / (both_model_rt['comp_time'].values + overheads)
        print model, percent_overheads
        all_percent_overheads.append(percent_overheads)

        both_model_rt['comp_time'] += sampling_model_rt['comp_time'].values
        speed_ups = lemp_model_rt['comp_time'].values / both_model_rt['comp_time'].values
        all_speedups.append(speed_ups)
        # print model
        data = pd.concat([both_model_rt, blocked_mm_model_rt, simdex_model_rt,
            lemp_model_rt, fexipro_model_rt, fexipro_si_model_rt])
        if len(data) == 0: continue
        max_runtime = sorted([
            lemp_model_rt['comp_time'].max(),
            simdex_model_rt['comp_time'].max(),
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
            edgecolor='black',
            ax=ax_arr[i])
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
        if int(i / nrows) == 3:
            ax_arr[i].set_xlabel('K')
        else:
            ax_arr[i].set_xlabel('')

        ax_arr[i].grid(True)
        ax_arr[i].set_title(LABEL_DICT[model], y=y_title)
        sns.despine()

    all_speedups = np.ravel(all_speedups)
    print 'Average speedup: ' + str(np.mean(all_speedups))
    print 'Min speedup: ' + str(np.min(all_speedups))
    print 'Max speedup: ' + str(np.max(all_speedups))

    all_overheads = np.ravel(all_overheads)
    print 'Average overhead: ' + str(np.mean(all_overheads))
    print 'Min overhead: ' + str(np.min(all_overheads))
    print 'Max overhead: ' + str(np.max(all_overheads))

    all_percent_overheads = np.ravel(all_percent_overheads)
    print 'Average percent overhead: ' + str(np.mean(all_percent_overheads))
    print 'Min percent overhead: ' + str(np.min(all_percent_overheads))
    print 'Max percent overhead: ' + str(np.max(all_percent_overheads))

    legend = ax_arr[0].legend(loc='upper center', bbox_to_anchor=bbox_to_anchor,
            bbox_transform=plt.gcf().transFigure, ncol=6)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    save_figure('f-u-plot', (legend,))
    plt.show()

def appetizer_bar_plots(simdex_df, lemp_df, blocked_mm_df, models, model_labels,
        K=1, bbox_to_anchor=(0,0,1,1), figsize=(15, 10)):

    assert(len(models) == len(model_labels))
    plt.figure(figsize=figsize)


    simdex_rt = simdex_df.sort_values(by='comp_time').groupby( ['model', 'K'], as_index=False).first()
    simdex_rt['algo'] = 'Our Index'

    blocked_mm_rt = blocked_mm_df[['model', 'K', 'comp_time']]
    blocked_mm_rt['algo'] = 'Blocked MM'

    lemp_rt = lemp_df[['model', 'K', 'comp_time']]
    lemp_rt['algo'] = 'LEMP'


    query_string = '(' + ('or '.join([('model == "%s"' % model) for model in models])) + ')'
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
    legend = plt.legend(loc='upper center', bbox_to_anchor=bbox_to_anchor,
            borderaxespad=0., ncol=3)

    start, end = plt.ylim()
    plt.xticks(plt.xticks()[0], model_labels)
    plt.ylim([start, max_runtime * 1.1])
    plt.minorticks_on()
    plt.xlabel('Model')
    plt.ylabel('Time (s)')
    plt.grid(True)
    # plt.title(LABEL_DICT[model], y=y_title)
    sns.despine()

    save_figure('appetizer-bar', (legend,))
    plt.show()


def rmse_and_reg_plots(blocked_mm_df, lemp_df, rmse_df, model_prefix, regs,
                        simdex_df=None, fexipro_df=None, K=1, fname=None,
                        figsize=(28, 6.5), bbox_to_anchor=(0,0,1,1), title=False,
                        y_title=-0.4, annotate=True, xy_text=(15, 150)):
    blocked_mm_rt = blocked_mm_df.query('K == %d' % K)[['model', 'comp_time']]
    lemp_rt = lemp_df.query('K == %d' % K)[['model', 'comp_time']]

    if simdex_df is not None:
        simdex_rt = simdex_df.query('K == %d' % K).sort_values(by='comp_time').groupby(
            ['model', 'K'], as_index=False).first()[['model', 'comp_time']]
    if fexipro_df is not None:
        fexipro_rt = fexipro_df.query('K == %d' % K)[['model', 'comp_time']]

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize)
    blocked_mm_data = []
    lemp_data = []
    if simdex_df is not None:
        simdex_data = []
    if fexipro_df is not None:
        fexipro_data = []
    rmse_data = []

    for reg in regs:
        if reg == 'gold-standard':
            model = model_prefix + '-' + reg
            reg = '1'
        else:
            model = model_prefix + '-reg-' + reg

        test_rmse = rmse_df.query('model == "%s"' %
                model)['test_rmse'].values[0]
        rmse_data.append(test_rmse)

        blocked_mm_result = blocked_mm_rt.query('model == "%s"' %
        model)['comp_time'].values[0]
        blocked_mm_data.append(blocked_mm_result)

        lemp_result = lemp_rt.query('model == "%s"' %
                model)['comp_time'].values[0]
        lemp_data.append(lemp_result)

        if simdex_df is not None:
            simdex_result = simdex_rt.query('model == "%s"' %
                    model)['comp_time'].values[0]
            simdex_data.append(simdex_result)

        if fexipro_df is not None:
            fexipro_result = fexipro_rt.query('model == "%s"' % model)['comp_time'].values[0]
            fexipro_data.append(fexipro_result)

    if fexipro_df is not None:
        max_runtime = min(fexipro_data) # Fexipro will be cut off, except for the smallest value
    else:
        max_runtime = max([max(blocked_mm_data), max(lemp_data),])

    min_rmse = min(rmse_data)
    best_reg_index = rmse_data.index(min_rmse)
    best_reg = regs[best_reg_index]

    ax1.plot(regs, rmse_data, color='black', marker='o')
    ax1.set_xscale('log')
    ax1.set_ylabel('Test RMSE')
    ax1.grid(True)
    ax1.minorticks_on()
    if annotate:
        ax1.annotate('best model',
                    xy=(best_reg, min_rmse*1.01), xycoords='data',
                    xytext=xy_text, textcoords='offset points',
                    arrowprops=dict(facecolor='black', shrink=0.03),
                    horizontalalignment='center', verticalalignment='bottom',
                    fontsize=30)


    min_rt = min([blocked_mm_data[best_reg_index], lemp_data[best_reg_index]])

    num_legend_entries = 2
    ax2.plot(regs, blocked_mm_data, label='Blocked MM', marker='o')
    if simdex_df is not None:
        num_legend_entries += 1
        ax2.plot(regs, simdex_data, label='SimDex\'s Index', marker='o')
    ax2.plot(regs, lemp_data, label='LEMP', marker='o')
    if fexipro_df is not None:
        num_legend_entries += 1
        ax2.plot(regs, fexipro_data, label='Fexipro', marker='o')
    #legend = ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if annotate:
        ax2.annotate('best runtime\non best model',
                    xy=(best_reg, min_rt*1.09), xycoords='data',
                    xytext=xy_text, textcoords='offset points',
                    arrowprops=dict(facecolor='black', shrink=0.03),
                    horizontalalignment='center', verticalalignment='bottom',
                    fontsize=30)

    if title:
        ax2.set_title(LABEL_DICT[model_prefix], y=y_title)
    sns.despine()
    ax2.set_xscale('log')
    ax2.set_xlabel(r'Regularization')
    ax2.set_ylabel('Time (s)')
    start, end = ax2.get_ylim()
    ax2.set_ylim([start, max_runtime * 1.1])
    ax2.grid(True)
    ax2.minorticks_on()

    #legend = plt.legend(loc='upper center', bbox_to_anchor=bbox_to_anchor,
    #        bbox_transform=plt.gcf().transFigure, ncol=num_legend_entries)
    #sup_title = fig.suptitle(title, y=y_title)
    fig.tight_layout()
    if fname:
        save_figure(fname)
    else:
        save_figure('rmse-reg-' + model_prefix)
    plt.show()

def rmse_and_reg_legend(add_simdex=False):
    # save legend separately
    fig = pylab.figure()
    figlegend = pylab.figure(figsize=(15, 1.5))
    ax = fig.add_subplot(111)
    if add_simdex:
        lines = ax.plot(range(10), pylab.randn(10), 'black', range(10), pylab.randn(10),
                range(10), pylab.randn(10), range(10), pylab.randn(10), marker='o')
        figlegend.legend(lines, ('Model Error', 'Blocked MM',
            'SimDex-Index Only', 'LEMP'), loc='center', ncol=4)
        figlegend.savefig(FIGURES_DIR + 'rmse-reg-legend.pdf')

    else:
        lines = ax.plot(range(10), pylab.randn(10), 'black',
                range(10), pylab.randn(10), range(10), pylab.randn(10), marker='o')
        figlegend.legend(lines, ('Model Error', 'Blocked MM', 'LEMP'), loc='center', ncol=4)
        figlegend.savefig(FIGURES_DIR + 'appetizer-legend.pdf')


def num_clusters_vs_runtime_single(simdex_df, model, nrows=1, filter_value=4096,
        figsize=(28, 6.5), bbox_to_anchor=(0,0,1,1), y_title=-0.25):

    num_legend_entries = -1

    table = simdex_df.query('model == "%s" and num_clusters < %d' % (model,
        filter_value))
    if len(table) == 0: return
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
        edgecolor='black')

    sns.despine()
    plt.title(model, y=y_title)
    plt.xlabel('K')
    plt.ylabel('Time (s)')
    plt.grid(True)
    plt.minorticks_on()

    legend = plt.legend(loc='upper left', bbox_to_anchor=bbox_to_anchor)
    modify_legend(legend, [('$C=', '$')])

    save_figure('n-clusters-vs-runtime-%s' % model, (legend,))
    plt.show()



def num_clusters_vs_runtime(simdex_df, models, nrows=1, min_value=0, max_value=4096,
        figsize=(28, 6.5), bbox_to_anchor=(0,0,1,1), y_title=-0.25):

    num_legend_entries = -1
    fig, ax_arr = plt.subplots(nrows=nrows, ncols=int(len(models) /
        nrows), figsize=figsize)

    for i, model in enumerate(models):
        table = simdex_df.query('model == "%s" and num_clusters < %d and num_clusters > %d' % (model, max_value, min_value))
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

    legend = ax_arr[0].legend(loc='upper center', bbox_to_anchor=bbox_to_anchor,
            bbox_transform=plt.gcf().transFigure, ncol=num_legend_entries)
    modify_legend(legend, [('$C=', '$')])

    save_figure('n-clusters-vs-runtime', (legend,))
    plt.show()

def batch_size_vs_runtime_single(simdex_df, model, figsize=(28, 6.5),
        bbox_to_anchor=(0,0,1,1), y_title=-0.25):
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

    legend = plt.legend(loc='upper center', bbox_to_anchor=bbox_to_anchor,
            bbox_transform=plt.gcf().transFigure)
    modify_legend(legend, [('Batch Size = ', '')])

    save_figure('batch-size-vs-runtime', (legend))
    plt.show()


###
# Not the same as num_clusters_vs_runtime: We find the best runtime and the
# associated num_clusters; then for that value of num_clusters, we get all
# the batch_sizes
###
def batch_size_vs_runtime(simdex_df, models, nrows=1, figsize=(28, 6.5),
        bbox_to_anchor=(0,0,1,1), y_title=-0.25):
    best_rt = simdex_df.sort_values(by='comp_time').groupby(
        ['model', 'K'], as_index=False).first()
    num_legend_entries = len(simdex_df['batch_size'].unique())
    fig, ax_arr = plt.subplots(nrows=nrows, ncols=int(len(models) / nrows),
            sharey=True, figsize=figsize)

    for i, model in enumerate(models):
        best_rt_model = best_rt.query('model == "%s"' % model)
        if len(best_rt_model) == 0: continue
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

    legend = plt.legend(loc='upper center', bbox_to_anchor=bbox_to_anchor,
            bbox_transform=plt.gcf().transFigure, ncol=num_legend_entries)
    modify_legend(legend, [('Batch Size = ', '')])

    save_figure('batch-size-vs-runtime', (legend))
    plt.show()


def num_bins_vs_runtime(simdex_df, models, nrows=1, figsize=(28, 6.5),
        bbox_to_anchor=(0,0,1,1), y_title=-0.25):
    best_rt = simdex_df.sort_values(by='comp_time').groupby(
        ['model', 'K'], as_index=False).first()
    num_legend_entries = len(best_rt['num_bins'].unique())
    fig, ax_arr = plt.subplots(nrows=nrows, ncols=int(len(models) / nrows),
            sharey=True, figsize=figsize)

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

    legend = plt.legend(loc='upper center', bbox_to_anchor=bbox_to_anchor,
            bbox_transform=plt.gcf().transFigure, ncol=num_legend_entries)
    modify_legend(legend, [('$B=', '$')])

    #fig.subplots_adjust(hspace=0)
    save_figure('n-bins-vs-runtime', (legend))
    plt.show()


def point_query_time(models, csv_dir='user-stats/K-1', sample_fraction=0.1,
        bins=10, nrows=1, figsize=(32, 4.5), y_title=-0.40):
    if nrows == 1:
        fig, ax_arr = plt.subplots(nrows=nrows, ncols=int(len(models) / nrows),
                sharey=True, figsize=figsize)
    else:
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 6) # hard-coded to 6 columns for now,
                                     # optimized for the 5-suplot case

        ax1 = plt.subplot(gs[0, 0:2])
        ax2 = plt.subplot(gs[0, 2:4])
        ax3 = plt.subplot(gs[0, 4:6])
        ax4 = plt.subplot(gs[1, 1:3])
        ax5 = plt.subplot(gs[1, 3:5])

        ax_arr = [ax1,ax2,ax3,ax4,ax5]

    for i, model in enumerate(models):
        for filename in glob.iglob('%s/%s_user_stats_*' % (csv_dir, model)):
            df = pd.read_csv(filename)
            x = df['query_time']
            print np.mean(x), len(x)
            sorted_x = sorted(x.sample(int(len(x)*sample_fraction)))
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


def reg_vs_runtime(simdex_df, blocked_mm_df, model_prefixes, regs, fname,
        nrows=1, figsize=(28, 6.5), y_title=-0.25):
    simdex_rt = simdex_df.sort_values(by='comp_time').groupby(
        ['model', 'K'], as_index=False).first()[['model', 'K', 'comp_time']]
    blocked_mm_rt = blocked_mm_df[['model', 'K', 'comp_time']]

    fig, ax_arr = plt.subplots(nrows=nrows, ncols=int(len(model_prefixes) /
        nrows), figsize=figsize)
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
    plot_cdf(vals, xlabel, x_line=x_line, y_line=y_line, title=title, log=log, fname=fname, show=True)


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
