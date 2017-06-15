#!/usr/bin/env bash

set -x

aws s3 cp s3://futuredata-simdex/stats/point-query-stats/ point-query-stats/ --recursive
aws s3 cp s3://futuredata-simdex/stats/decision-rule-with-K/ decision-rule-with-K/ --recursive
