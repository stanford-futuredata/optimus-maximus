#!/usr/bin/env python

from daal.data_management import (
    CSRNumericTable,
    NumericTableIface,
    readOnly,
    readWrite,
    BlockDescriptor,
    packed_mask,
    HomogenNumericTable,
    KeyValueDataCollection,
    DataSourceIface,
    FileDataSource,
    TensorIface,
    HomogenTensor,
    SubtensorDescriptor,
    HomogenNumericTable_Float32, )

from daal.services import Collection

import sys
import numpy as np
import random as rand

if sys.version[0] == '2':
    import Queue as Queue
else:
    import queue as Queue

import daal.algorithms.kmeans as kmeans
from daal.algorithms.kmeans import init
import time
from daal.services import Environment

from daal_utils import printNumericTable

if len(sys.argv) != 5:
    print("args: NumClusters NumIterations OutputDir SampleSize%")
    sys.exit(0)

# nClusters = 1024
# nIterations = 25
# nThreads = 1

nClusters = int(sys.argv[1])
nIterations = int(sys.argv[2])
nThreads = 1
input_dir = sys.argv[3]
sampleSize = int(sys.argv[4])

# Get the number of threads that is used by the library by default
nThreadsInit = Environment.getInstance().getNumberOfThreads()
print(nThreadsInit)

# Set the maximum number of threads to be used by the library
Environment.getInstance().setNumberOfThreads(nThreads)

# Get the number of threads that is used by the library after changing
nThreadsNew = Environment.getInstance().getNumberOfThreads()
print("max number of threads: %d" % nThreadsNew)

datasetFileName = input_dir + "/s2_userWeights.csv"

cluster_dir = input_dir + '/' + str(sampleSize) + '/' + str(nIterations)

centroidsFileName = cluster_dir + "/" + str(nClusters) + "_centroids.csv"

centroidSource = FileDataSource(centroidsFileName,
                                DataSourceIface.doAllocateNumericTable,
                                DataSourceIface.doDictionaryFromContext)
centroidSource.loadDataBlock()

print(input_dir)

t0 = time.time()
dataSource = FileDataSource(datasetFileName,
                            DataSourceIface.doAllocateNumericTable,
                            DataSourceIface.doDictionaryFromContext)

dataSource.loadDataBlock()

initAlg = init.Batch_Float32DeterministicDense(nClusters)
initAlg.input.set(init.data, centroidSource.getNumericTable())

t1 = time.time()
init_time = t1 - t0
print("init time: %f" % init_time)

t0 = time.time()
res = initAlg.compute()
t1 = time.time()
centroid_time = t1 - t0
print("centroid time: %f" % centroid_time)

centroidsResult = res.get(init.centroids)

algorithm = kmeans.Batch_Float32LloydDense(nClusters, 0)
algorithm.input.set(kmeans.data, dataSource.getNumericTable())
algorithm.input.set(kmeans.inputCentroids, centroidsResult)

t0 = time.time()
res2 = algorithm.compute()
t1 = time.time()
cluster_time = t1 - t0
print("cluster time: %f" % cluster_time)

printNumericTable(
    res2.get(kmeans.centroids), "First 10 dimensions of centroids:", 20, 10)

assignments_table = res2.get(kmeans.assignments)
assignment_num_rows = assignments_table.getNumberOfRows()
assignment_num_cols = assignments_table.getNumberOfColumns()

assignments_block = BlockDescriptor()
assignments_table.getBlockOfRows(0, assignment_num_rows, readOnly,
                                 assignments_block)
# assignments numpy array
assignments_array = assignments_block.getArray()

centroids_table = res2.get(kmeans.centroids)
centroids_num_rows = centroids_table.getNumberOfRows()
centroids_num_cols = centroids_table.getNumberOfColumns()

centroids_block = BlockDescriptor()
centroids_table.getBlockOfRows(0, centroids_num_rows, readOnly,
                               centroids_block)
# centroids numpy array
centroids_array = centroids_block.getArray()

# create user_cluster_ids file
# user_ids = np.loadtxt("/Users/geetsethi/Library/Developer/Xcode/DerivedData/fomo_preproc-fuujyptqvyyskicvhnldhaqelavt/Build/Products/Release/ids_userWeight.txt", dtype=int)
pathnm = input_dir + "/ids_userWeight.txt"
# pathnm = sys.argv[5]

user_ids = np.loadtxt(pathnm, dtype=int)

# user_ids_sampled = user_ids[random_indices]

if user_ids.shape[0] != assignments_array.shape[0]:
    print("user_ids - assignments array mismatch")

t0 = time.time()

user_to_clusters_filename = cluster_dir + '/' + str(
    nClusters) + "_user_cluster_ids"
with open(user_to_clusters_filename, 'w') as f:
    for i in range(user_ids.shape[0]):
        print("%d: %d" % (user_ids[i], int(assignments_array[i][0])), file=f)

clusters_filename = cluster_dir + '/' + str(nClusters) + "_user_clusters"
with open(clusters_filename, 'w') as f2:
    for i in range(centroids_array.shape[0]):
        factor_str = ','.join(str(factor) for factor in centroids_array[i])
        f2.write('(%d,%s)\n' % (i, factor_str))

import re
from collections import defaultdict

CLUSTER_ID_REGEX = r'(\d+): (\d+)'


def invert_cluster_id_file(filename):
    with open(filename) as r:
        cluster_id_user_id_dict = defaultdict(list)
        for line in r:
            m = re.match(CLUSTER_ID_REGEX, line)
            user_id, cluster_id = int(m.group(1)), int(m.group(2))
            cluster_id_user_id_dict[cluster_id].append(user_id)
    with open(filename + '_inverted', 'w') as inverted_outfile, \
            open(filename + '_counts', 'w') as counts_outfile:
        for cluster_id, user_ids in cluster_id_user_id_dict.items():
            # print >> inverted_outfile, '%d: [%s]' % (cluster_id, ', '.join(str(u) for u in user_ids))
            inverted_outfile.write('%d: [%s]\n' %
                                   (cluster_id,
                                    ', '.join(str(u) for u in user_ids)))
            # print >> counts_outfile, '%d,%d' % (cluster_id, len(user_ids))
            counts_outfile.write('%d,%d\n' % (cluster_id, len(user_ids)))


invert_cluster_id_file(user_to_clusters_filename)

t1 = time.time()
output_time = t1 - t0
print("output time: %f" % output_time)

cluster_time_filename = "cluster_time_u" + str(user_ids.shape[0]) + "_f" + str(
    centroids_array.shape[1]) + "_c" + str(nClusters) + ".txt"
with open(cluster_time_filename, 'a') as f:
    output = "Second pass cluster time: " + str(
        cluster_time)[:5] + " Total time: " + str(
            centroid_time + cluster_time)[:5] + " Output Time: " + str(
                output_time)[:5] + "\n"
    f.write(output)

    # from daal.data_management import NumericTable
#
# NumericTable.