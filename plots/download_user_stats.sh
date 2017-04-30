#!/usr/bin/env bash

set -x

aws s3 cp s3://ocius-experiments/user-stats/ user-stats/ --recursive
