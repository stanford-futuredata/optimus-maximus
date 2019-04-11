#! /usr/bin/env python

import pandas as pd
import numpy as np
import argparse


# Superclass for calculating stats on decision rule for any index
class IndexDecisionRuleStats:

    BLOCKED_MM_COL = 'blocked_mm_sample_time'
    SAMPLED_USERS_COL = 'num_sampled_users'

    def __init__(self, sample_time_col, dec_rule_df, lemp_truth_df,
                 index_truth_df, blocked_mm_truth_df):
        self.sample_time_col = sample_time_col
        self.dec_rule_df = dec_rule_df
        self.lemp_truth_df = lemp_truth_df
        self.index_truth_df = index_truth_df
        self.blocked_mm_truth_df = blocked_mm_truth_df
        self.results = {
            'model': [],
            'K': [],
            'blocked_mm_sample_time': [],
            'index_sample_time': [],
            'index_true_rt': [],
            'index_estimate_rt': [],
            'blocked_mm_true_rt': [],
            'blocked_mm_estimate_rt': [],
            'correct': [],
            'overhead_perc': [],
        }
        self.stats = {
            'index_vs_blocked_mm': [],
            'index_vs_lemp': [],
            'blockopt_vs_oracle': [],
            'blockopt_vs_lemp': [],
            'oracle_vs_lemp': []
        }
        self.results_df = None
        self.stats_df = None

    def num_users(self, model):
        if 'Netflix' in model:
            return 480189
        elif 'KDD' in model:
            return 1000990
        elif 'R2' in model:
            return 1823179
        elif 'Glove' in model:
            return 100000

    def estimate_blocked_mm_runtime(self, row):
        return row[self.BLOCKED_MM_COL] * row[self.SAMPLED_USERS_COL]

    def blocked_mm_true_runtime(self, model, K):
        return self.blocked_mm_truth_df.query('model == "%s" and K == %d' %
                                              (model, K))['comp_time'].min()

    def lemp_true_runtime(self, model, K):
        return self.lemp_truth_df.query('model == "%s" and K == %d' %
                                        (model, K))['comp_time'].min()

    def optimizer_runtime(self, row):
        overhead_rt = self.overhead_runtime(row)
        optimizer_rt = self.optimizer_runtime_no_overhead(row)
        return optimizer_rt + overhead_rt

    def oracle_runtime(self, row):
        index_true_rt = self.index_true_runtime(row)
        blocked_mm_true_rt = self.blocked_mm_true_runtime(
            row['model'], row['K'])
        return min(index_true_rt, blocked_mm_true_rt)

    def add_row_to_results_df(self, row):
        model, K = row['model'], row['K']
        correct = self.correct(row)

        self.results['model'].append(model)
        self.results['K'].append(K)
        self.results['blocked_mm_sample_time'].append(row[self.BLOCKED_MM_COL])
        self.results['index_sample_time'].append(row[self.sample_time_col])
        self.results['correct'].append(correct)

        overhead_rt = self.overhead_runtime(row)
        optimizer_rt = self.optimizer_runtime(row)
        overhead_perc = overhead_rt / optimizer_rt

        self.results['overhead_perc'].append(overhead_perc)
        self.results['index_true_rt'].append(self.index_true_runtime(row))
        self.results['index_estimate_rt'].append(
            self.estimate_index_runtime(row))

        self.results['blocked_mm_true_rt'].append(
            self.blocked_mm_true_runtime(model, K))
        self.results['blocked_mm_estimate_rt'].append(
            self.estimate_blocked_mm_runtime(row))

    def add_row_to_stats_df(self, row):
        model, K = row['model'], row['K']
        oracle_rt = self.oracle_runtime(row)
        optimizer_rt = self.optimizer_runtime(row)
        index_true_rt = self.index_true_runtime(row)
        lemp_true_rt = self.lemp_true_runtime(model, K)
        blocked_mm_true_rt = self.blocked_mm_true_runtime(model, K)

        self.stats['index_vs_blocked_mm'].append(
            blocked_mm_true_rt / index_true_rt)
        self.stats['index_vs_lemp'].append(lemp_true_rt / index_true_rt)
        self.stats['blockopt_vs_oracle'].append(oracle_rt / optimizer_rt)
        self.stats['blockopt_vs_lemp'].append(lemp_true_rt / optimizer_rt)
        self.stats['oracle_vs_lemp'].append(lemp_true_rt / oracle_rt)

    def generate_results_and_stats(self):
        for _, row in self.dec_rule_df.iterrows():
            self.add_row_to_results_df(row)
            self.add_row_to_stats_df(row)

        self.results_df = pd.DataFrame.from_dict(self.results)
        self.stats_df = pd.DataFrame.from_dict(self.stats)

        return self.results_df, self.stats_df

    # Need to be implemented by subclass
    def correct(self, row):
        pass

    def overhead_runtime(self, row):
        pass

    def preproc_runtime(self, row):
        pass

    def estimate_index_runtime(self, row):
        pass

    def index_true_runtime(self, row):
        pass

    def optimizer_runtime_no_overhead(self, row):
        pass


# Stats for LEMP Decision Rule
class LempDecisionRuleStats(IndexDecisionRuleStats):
    def __init__(self, sample_time_col, dec_rule_df, lemp_truth_df,
                 index_truth_df, blocked_mm_truth_df):
        super(LempDecisionRuleStats,
              self).__init__(sample_time_col, dec_rule_df, lemp_truth_df,
                             index_truth_df, blocked_mm_truth_df)

    # Subclass methods that don't have implementations
    def correct(self, row):
        return (self.blocked_mm_true_runtime(row['model'], row['K']) >
                self.index_true_runtime(row)) == row['lemp_wins']

    def overhead_runtime(self, row):
        return row[self.BLOCKED_MM_COL] * row['num_users'] if row['lemp_wins'] \
                else self.preproc_runtime(row) + row[self.sample_time_col] * row['num_users']

    def preproc_runtime(self, row):
        return row['preproc_time'] + row['index_time']

    def estimate_index_runtime(self, row):
        return row[self.sample_time_col] * self.num_users(row['model'])

    def index_true_runtime(self, row):
        return self.lemp_true_runtime(row['model'], row['K'])

    def optimizer_runtime_no_overhead(self, row):
        return self.index_true_runtime(row) if row[
            'lemp_wins'] else self.blocked_mm_true_runtime(
                row['model'], row['K'])

    # Subclass this method, too, since LEMP output files don't have 'num_sampled_users' column
    def estimate_blocked_mm_runtime(self, row):
        return row[self.BLOCKED_MM_COL] * row['num_users']


# Stats for SimpleBaseline Decision Rule
class SimpleBaselineDecisionRuleStats(IndexDecisionRuleStats):
    def __init__(self, sample_time_col, dec_rule_df, lemp_truth_df,
                 index_truth_df, blocked_mm_truth_df):
        super(SimpleBaselineDecisionRuleStats, self).__init__(
            sample_time_col, dec_rule_df, lemp_truth_df, index_truth_df,
            blocked_mm_truth_df)

    # Subclass methods that don't have implementations
    def correct(self, row):
        return (self.blocked_mm_true_runtime(row['model'], row['K']) >
                self.index_true_runtime(row)) == row['simdex_wins']

    def overhead_runtime(self, row):
        return row[self.BLOCKED_MM_COL] * row[self.SAMPLED_USERS_COL] if row['simdex_wins'] \
                else self.preproc_runtime(row) + row[self.sample_time_col] * \
                     row[self.SAMPLED_USERS_COL]

    def preproc_runtime(self, row):
        return row['index_time'] + row['cluster_time']

    def estimate_index_runtime(self, row):
        return row[self.sample_time_col] * self.num_users(row['model'])

    def index_true_runtime(self, row):
        return self.index_truth_df.query('model == "%s" and K == %d' %
                                         (row['model'],
                                          row['K']))['comp_time'].min()

    def optimizer_runtime_no_overhead(self, row):
        return self.index_true_runtime(row) if row[
            'simdex_wins'] else self.blocked_mm_true_runtime(
                row['model'], row['K'])


# Stats for Fexipro Decision Rule
class FexiproDecisionRuleStats(IndexDecisionRuleStats):
    def __init__(self, sample_time_col, dec_rule_df, lemp_truth_df,
                 index_truth_df, blocked_mm_truth_df):
        super(FexiproDecisionRuleStats,
              self).__init__(sample_time_col, dec_rule_df, lemp_truth_df,
                             index_truth_df, blocked_mm_truth_df)

    # Subclass methods that don't have implementations
    def correct(self, row):
        return (self.blocked_mm_true_runtime(row['model'], row['K']) >
                self.index_true_runtime(row)) == row['fexipro_wins']

    def overhead_runtime(self, row):
        return row[self.BLOCKED_MM_COL] * row[self.SAMPLED_USERS_COL] if row['fexipro_wins'] \
                else self.preproc_runtime(row) + row[self.sample_time_col] * \
                     row[self.SAMPLED_USERS_COL]

    def preproc_runtime(self, row):
        return row['preproc_time']

    def estimate_index_runtime(self, row):
        return row[self.sample_time_col] * self.num_users(row['model'])

    def index_true_runtime(self, row):
        return self.index_truth_df.query(
            'model == "%s" and K == %d and alg == "%s"' %
            (row['model'], row['K'], row['alg']))['comp_time'].min()

    def optimizer_runtime_no_overhead(self, row):
        return self.index_true_runtime(row) if row[
            'fexipro_wins'] else self.blocked_mm_true_runtime(
                row['model'], row['K'])


# Stats for Lemp + SB Decision Rule
class LempSimpleBaselineDecisionRuleStats(IndexDecisionRuleStats):
    BLOCKED_MM = 0
    SB = 1
    LEMP = 2

    def __init__(self, lemp_sample_time_col, sample_time_col, lemp_dec_rule_df,
                 dec_rule_df, lemp_truth_df, index_truth_df,
                 blocked_mm_truth_df):
        super(LempSimpleBaselineDecisionRuleStats, self).__init__(
            sample_time_col, dec_rule_df, lemp_truth_df, index_truth_df,
            blocked_mm_truth_df)
        self.lemp_sample_time_col = lemp_sample_time_col
        self.lemp_dec_rule_df = lemp_dec_rule_df

    def lemp_dec_rule_row(self, model, K):
        single_row = self.lemp_dec_rule_df.query('model == "%s" and K == %d' %
                                                 (model, K))
        assert (len(single_row) == 1)
        return single_row.iloc[0]

    def decision_rule_winner(self, row):
        blocked_mm_sample_time = row[self.BLOCKED_MM_COL]
        simdex_sample_time = row[self.sample_time_col]
        lemp_sample_time = self.lemp_dec_rule_row(
            row['model'], row['K'])[self.lemp_sample_time_col].min()

        return self.BLOCKED_MM if min(blocked_mm_sample_time, simdex_sample_time, lemp_sample_time) \
                   == blocked_mm_sample_time else \
               self.LEMP if min(blocked_mm_sample_time, simdex_sample_time, lemp_sample_time) \
                   == lemp_sample_time else \
               self.SB

    def actual_winner(self, row):
        blocked_mm_true_rt = self.blocked_mm_true_runtime(
            row['model'], row['K'])
        simdex_true_rt = self.index_true_runtime(row)
        lemp_true_rt = self.lemp_true_runtime(row['model'], row['K'])

        return self.BLOCKED_MM if min(blocked_mm_true_rt, simdex_true_rt, lemp_true_rt) == blocked_mm_true_rt else \
               self.LEMP if min(blocked_mm_true_rt, simdex_true_rt, lemp_true_rt) == lemp_true_rt else \
               self.SB

    # Subclass this, too, to pick the winner, amongst all three
    def oracle_runtime(self, row):
        model, K = row['model'], row['K']
        index_true_rt = self.index_true_runtime(row)
        blocked_mm_true_rt = self.blocked_mm_true_runtime(model, K)
        return min(index_true_rt, blocked_mm_true_rt,
                   self.lemp_true_runtime(model, K))

    # Subclass methods that don't have implementations
    def correct(self, row):
        return (self.decision_rule_winner(row) == self.actual_winner(row))

    def overhead_runtime(self, row):
        lemp_dec_row = self.lemp_dec_rule_row(row['model'], row['K'])

        simdex_preproc_rt = self.preproc_runtime(row)
        lemp_preproc_rt = lemp_dec_row['preproc_time'] + lemp_dec_row['index_time']
        blocked_mm_overhead = row[self.BLOCKED_MM_COL] * row[
            self.SAMPLED_USERS_COL]
        simdex_overhead = row[self.sample_time_col] * row[
            self.SAMPLED_USERS_COL]
        lemp_overhead = lemp_dec_row[self.lemp_sample_time_col] * lemp_dec_row[
            'num_users']

        if self.decision_rule_winner(row) == self.LEMP:
            return simdex_preproc_rt + blocked_mm_overhead + simdex_overhead
        elif self.decision_rule_winner(row) == self.SB:
            return lemp_preproc_rt + blocked_mm_overhead + lemp_overhead
        else:  # BLOCKED_MM
            return simdex_preproc_rt + simdex_overhead + lemp_preproc_rt + lemp_overhead

    def preproc_runtime(self, row):
        return row['index_time'] + row['cluster_time']

    def estimate_index_runtime(self, row):
        return row[self.sample_time_col] * self.num_users(row['model'])

    def index_true_runtime(self, row):
        return self.index_truth_df.query('model == "%s" and K == %d' %
                                         (row['model'],
                                          row['K']))['comp_time'].min()

    def optimizer_runtime_no_overhead(self, row):
        winner = self.decision_rule_winner(row)
        if winner == self.LEMP:
            return self.lemp_true_runtime(row['model'], row['K'])
        elif winner == self.SB:
            return self.index_true_runtime(row)
        else:  # BLOCKED_MM
            return self.blocked_mm_true_runtime(row['model'], row['K'])


# Flags for command-line args
RUN_LEMP = 0
RUN_SB = 1
RUN_FEXIPRO_SI = 2
RUN_FEXIPRO_SIR = 3
RUN_LEMP_SB = 4


def print_stats(results_df, stats_df):
    print(results_df.query('correct == False'))
    print('Accuracy:', np.mean(results_df['correct']))
    print('Percent Overhead, Avg/Std.Dev/Min/Max:',
          np.mean(results_df['overhead_perc']),
          np.std(results_df['overhead_perc']),
          np.min(results_df['overhead_perc']),
          np.max(results_df['overhead_perc']))

    print('Index vs Blocked MM: Avg/Std.Dev/Min/Max:',
          np.mean(stats_df['index_vs_blocked_mm']),
          np.std(stats_df['index_vs_blocked_mm']),
          np.min(stats_df['index_vs_blocked_mm']),
          np.max(stats_df['index_vs_blocked_mm']),
          np.mean(stats_df['index_vs_blocked_mm'] > 1.0))
    print('Index vs Lemp, Avg/Std.Dev/Min/Max, % Faster:',
          np.mean(stats_df['index_vs_lemp']),
          np.std(stats_df['index_vs_lemp']),
          np.min(stats_df['index_vs_lemp']),
          np.max(stats_df['index_vs_lemp']),
          np.mean(stats_df['index_vs_lemp'] > 1.0))

    print('RecOpt vs Lemp, Avg/Std.Dev/Min/Max, % Faster:',
          np.mean(stats_df['blockopt_vs_lemp']),
          np.std(stats_df['blockopt_vs_lemp']),
          np.min(stats_df['blockopt_vs_lemp']),
          np.max(stats_df['blockopt_vs_lemp']))
    print('RecOpt vs Oracle, Avg/Min/Max:',
          np.mean(stats_df['blockopt_vs_oracle']),
          np.min(stats_df['blockopt_vs_oracle']),
          np.max(stats_df['blockopt_vs_oracle']))
    print('Oracle vs Lemp:', np.mean(stats_df['oracle_vs_lemp']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--lemp', dest='index_type', action='store_const', const=RUN_LEMP)
    parser.add_argument(
        '--sb', dest='index_type', action='store_const', const=RUN_SB)
    parser.add_argument(
        '--fexipro-si',
        dest='index_type',
        action='store_const',
        const=RUN_FEXIPRO_SI)
    parser.add_argument(
        '--fexipro-sir',
        dest='index_type',
        action='store_const',
        const=RUN_FEXIPRO_SIR)
    parser.add_argument(
        '--lemp-sb',
        dest='index_type',
        action='store_const',
        const=RUN_LEMP_SB)
    parser.set_defaults(index_type=RUN_LEMP)

    args = parser.parse_args()

    lemp_truth_df = pd.read_csv(
        '../plots/timing-results/lemp-gold-standard-timing.csv')
    blocked_mm_truth_netflix = pd.read_csv(
        '../plots/timing-results/netflix-blocked_mm-timing.csv')
    blocked_mm_truth_kdd = pd.read_csv(
        '../plots/timing-results/kdd-blocked_mm-timing.csv')
    blocked_mm_truth_r2 = pd.read_csv(
        '../plots/timing-results/r2-blocked_mm-timing.csv')
    blocked_mm_truth_glove = pd.read_csv(
        '../plots/timing-results/glove-blocked_mm-timing.csv')
    blocked_mm_truth_df = pd.concat([
        blocked_mm_truth_netflix, blocked_mm_truth_kdd, blocked_mm_truth_r2,
        blocked_mm_truth_glove
    ])

    if args.index_type == RUN_LEMP:
        lemp_dec_rule_df = pd.read_csv('lemp-decision-rule.csv')
        decision_rule_stats = LempDecisionRuleStats(
            'lemp_sample_time', lemp_dec_rule_df, lemp_truth_df, lemp_truth_df,
            blocked_mm_truth_df)
        results_df, stats_df = decision_rule_stats.generate_results_and_stats()
    elif args.index_type == RUN_SB or args.index_type == RUN_LEMP_SB:

        simdex_truth_netflix = pd.read_csv(
            '../plots/timing-results/netflix-simdex-timing.csv')
        simdex_truth_kdd = pd.read_csv(
            '../plots/timing-results/kdd-simdex-timing.csv')
        simdex_truth_r2 = pd.read_csv(
            '../plots/timing-results/r2-simdex-timing.csv')
        simdex_truth_glove = pd.read_csv(
            '../plots/timing-results/glove-simdex-timing.csv')
        simdex_truth_df = pd.concat([
            simdex_truth_netflix, simdex_truth_kdd, simdex_truth_r2,
            simdex_truth_glove
        ])
        simdex_dec_rule_df = pd.read_csv('simdex-decision-rule.csv')

        if args.index_type == RUN_SB:

            decision_rule_stats = SimpleBaselineDecisionRuleStats(
                'simdex_sample_time', simdex_dec_rule_df, lemp_truth_df,
                simdex_truth_df, blocked_mm_truth_df)
            results_df, stats_df = decision_rule_stats.generate_results_and_stats(
            )

        else:  # LEMP_SB
            lemp_dec_rule_df = pd.read_csv('lemp-decision-rule.csv')

            decision_rule_stats = LempSimpleBaselineDecisionRuleStats(
                'lemp_sample_time', 'simdex_sample_time', lemp_dec_rule_df,
                simdex_dec_rule_df, lemp_truth_df, simdex_truth_df,
                blocked_mm_truth_df)
            results_df, stats_df = decision_rule_stats.generate_results_and_stats(
            )

    else:  # FEXIPRO
        alg = 'SI' if args.index_type == RUN_FEXIPRO_SI else 'SIR'
        fexipro_dec_rule_df = pd.read_csv('fexipro-decision-rule.csv').query(
            'alg == "%s"' % alg)
        fexipro_truth_df = pd.read_csv(
            '../plots/timing-results/fexipro-orig-timing.csv').query(
                'alg == "%s"' % alg)

        decision_rule_stats = FexiproDecisionRuleStats(
            'fexipro_sample_time', fexipro_dec_rule_df, lemp_truth_df,
            fexipro_truth_df, blocked_mm_truth_df)
        results_df, stats_df = decision_rule_stats.generate_results_and_stats()

    print_stats(results_df, stats_df)


if __name__ == '__main__':
    main()
