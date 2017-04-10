#!/usr/bin/env python

import pandas as pd

df = pd.read_csv('simdex-timing.csv')
simdex_lemp_netflix_noav_10 = df.query(
    'model == "simdex-timing/lemp-paper-Netflix-noav-10" and num_threads == 1')
simdex_lemp_netflix_noav_50 = df.query(
    'model == "simdex-timing/lemp-paper-Netflix-noav-50" and num_threads == 1')
simdex_lemp_netflix_noav_100 = df.query(
    'model == "simdex-timing/lemp-paper-Netflix-noav-100" and num_threads == 1')
simdex_lemp_netflix_50 = df.query(
    'model == "simdex-timing/lemp-paper-Netflix-50" and num_threads == 1')
simdex_pb_new_netflix_10 = df.query(
    'model == "simdex-timing/pb-new-Netflix-10" and num_threads == 1')
simdex_pb_new_netflix_25 = df.query(
    'model == "simdex-timing/pb-new-Netflix-25" and num_threads == 1')
simdex_pb_new_netflix_50 = df.query(
    'model == "simdex-timing/pb-new-Netflix-50" and num_threads == 1')
simdex_sigmod_netflix_10 = df.query(
    'model == "simdex-timing/sigmod-deadline-Netflix-10" and num_threads == 1')
simdex_sigmod_netflix_25 = df.query(
    'model == "simdex-timing/sigmod-deadline-Netflix-25" and num_threads == 1')
simdex_sigmod_netflix_50 = df.query(
    'model == "simdex-timing/sigmod-deadline-Netflix-50" and num_threads == 1')

df2 = pd.read_csv('lemp-timing.csv')
lemp_lemp_netflix_noav_10 = df2.query(
    'model == "lemp-timing/lemp-paper-Netflix-noav-10" and num_threads == 1')
lemp_lemp_netflix_noav_50 = df2.query(
    'model == "lemp-timing/lemp-paper-Netflix-noav-50" and num_threads == 1')
lemp_lemp_netflix_noav_100 = df2.query(
    'model == "lemp-timing/lemp-paper-Netflix-noav-100" and num_threads == 1')
lemp_lemp_netflix_50 = df2.query(
    'model == "lemp-timing/lemp-paper-Netflix-50" and num_threads == 1')
lemp_pb_new_netflix_10 = df2.query(
    'model == "lemp-timing/pb-new-Netflix-10" and num_threads == 1')
lemp_pb_new_netflix_25 = df2.query(
    'model == "lemp-timing/pb-new-Netflix-25" and num_threads == 1')
lemp_pb_new_netflix_50 = df2.query(
    'model == "lemp-timing/pb-new-Netflix-50" and num_threads == 1')
lemp_sigmod_netflix_10 = df2.query(
    'model == "lemp-timing/sigmod-deadline-Netflix-10" and num_threads == 1')
lemp_sigmod_netflix_25 = df2.query(
    'model == "lemp-timing/sigmod-deadline-Netflix-25" and num_threads == 1')
lemp_sigmod_netflix_50 = df2.query(
    'model == "lemp-timing/sigmod-deadline-Netflix-50" and num_threads == 1')

