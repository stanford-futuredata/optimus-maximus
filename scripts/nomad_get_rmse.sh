#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
    echo "One argument required: [nomad output file]"
    exit 1
fi

OUTPUT_FILE=$1

set -x

echo "dummy,num_tasks,num_threads,timeout,global_num_updates,test_rmse,global_test_sum_error,global_test_count_error,global_num_failures,global_col_empty,global_send_count,train_rmse,,global_train_sum_error,global_train_count_error" > ${OUTPUT_FILE}.csv
grep "^testgrep," $OUTPUT_FILE >> ${OUTPUT_FILE}.csv
