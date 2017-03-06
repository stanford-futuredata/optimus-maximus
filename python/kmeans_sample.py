# from daal.data_management import FileDataSource, DataSourceIface
from daal.data_management import (
    CSRNumericTable, NumericTableIface, readOnly, readWrite, BlockDescriptor,
    packed_mask, HomogenNumericTable, KeyValueDataCollection,
    DataSourceIface, FileDataSource, TensorIface, HomogenTensor, SubtensorDescriptor,
    HomogenNumericTable_Float32,
)

from daal.services import Collection

import sys
import numpy as np
import random as rand
import math
import os


if sys.version[0] == '2':
    import Queue as Queue
else:
    import queue as Queue

import daal.algorithms.kmeans as kmeans
from daal.algorithms.kmeans import init
import time
from daal.services import Environment

# sys.path.append("/opt/intel/compilers_and_libraries_2017.0.102/mac/daal/examples/python/source")
from utils import printNumericTable

if len(sys.argv) != 5:
    print("args: NumClusters NumIterations OutputDir SampleSize%")
    sys.exit(0)

# nClusters = 1024
# nIterations = 25
# nThreads = 1

def mkdir_p(dir):
  if not os.path.exists(dir):
      os.makedirs(dir)

nClusters = int(sys.argv[1])
nIterations = int(sys.argv[2])
nThreads = 1
input_dir = sys.argv[3]
sampleSize = int(sys.argv[4])

if sampleSize > 100:
    print("Sample Size much be between 0 and 100")
    sys.exit(0)


# Get the number of threads that is used by the library by default
nThreadsInit = Environment.getInstance().getNumberOfThreads()
print(nThreadsInit)

# Set the maximum number of threads to be used by the library
Environment.getInstance().setNumberOfThreads(nThreads)

# Get the number of threads that is used by the library after changing
nThreadsNew = Environment.getInstance().getNumberOfThreads()
print("max number of threads: %d" % nThreadsNew)

# datasetFileName = "/Users/geetsethi/Library/Developer/Xcode/DerivedData/fomo_preproc-fuujyptqvyyskicvhnldhaqelavt/Build/Products/Release/s2_userWeights.csv"
datasetFileName = input_dir + "/s2_userWeights.csv"
# datasetFileName = sys.argv[4]

input_data = np.loadtxt(open(datasetFileName, "rb"), dtype=np.float32, delimiter=",")

if sampleSize == 0:
    samples = int(math.floor(input_data.shape[0]/float(5)))
    sampleSize = 20
else:
    samples = int(math.floor(input_data.shape[0]*(sampleSize/float(100))))


random_indices = rand.sample(range(input_data.shape[0]), samples)

sampled_data = input_data[random_indices]


print(input_dir)

t0 = time.time()
# dataSource = FileDataSource(datasetFileName, DataSourceIface.doAllocateNumericTable, DataSourceIface.doDictionaryFromContext)
#
# dataSource.loadDataBlock()

sample_data_table = HomogenNumericTable_Float32(sampled_data)


initAlg = init.Batch_Float32RandomDense(nClusters)
# initAlg.input.set(init.data, dataSource.getNumericTable())
initAlg.input.set(init.data, sample_data_table)


t1 = time.time()
init_time = t1-t0
print("init time: %f" % init_time)

t0 = time.time()
res = initAlg.compute()
t1 = time.time()
centroid_time = t1-t0
print("centroid time: %f" % centroid_time)

centroidsResult = res.get(init.centroids)

algorithm = kmeans.Batch_Float32LloydDense(nClusters, nIterations)
# algorithm.input.set(kmeans.data, dataSource.getNumericTable())
algorithm.input.set(kmeans.data, sample_data_table)
algorithm.input.set(kmeans.inputCentroids, centroidsResult)

t0 = time.time()
res2 = algorithm.compute()
t1 = time.time()
cluster_time = t1-t0
print("cluster time: %f" % cluster_time)

printNumericTable(res2.get(kmeans.centroids), "First 10 dimensions of centroids:", 20, 10)
# printNumericTable(res2.get(kmeans.goalFunction), "Goal function value:")


centroids_table = res2.get(kmeans.centroids)
centroids_num_rows = centroids_table.getNumberOfRows()
centroids_num_cols = centroids_table.getNumberOfColumns()

centroids_block = BlockDescriptor()
centroids_table.getBlockOfRows(0, centroids_num_rows, readOnly, centroids_block)
# centroids numpy array
centroids_array = centroids_block.getArray()



# clusters_filename = input_dir + '/' + str(nClusters)+ "_user_clusters"
# with open(clusters_filename, 'w') as f2:
#     for i in range(centroids_array.shape[0]):
#         factor_str = ','.join(str(factor) for factor in centroids_array[i])
#         f2.write('(%d,%s)\n' % (i, factor_str))


cluster_dir = input_dir + '/' + str(sampleSize) + '/' + str(nIterations)
mkdir_p(cluster_dir)

centroids_filename = cluster_dir + '/' + str(nClusters) + "_centroids.csv"
np.savetxt(centroids_filename, centroids_array, delimiter=",")

cluster_time_filename = "cluster_time_u" + str(input_data.shape[0]) + "_f" + str(input_data.shape[1]) + "_c" + str(nClusters) + ".txt"

with open(cluster_time_filename, 'w') as f:
    header = "FirstPassClusterTime,TotalTime,Sample%,Samples\n"
    f.write(header)
    # output = "First pass cluster time: " + str(cluster_time)[:5] + " Total time: " + \
    #          str(init_time+centroid_time+cluster_time)[:5] + " Samples %: " + str(sampleSize) + \
    #          " Samples: " + str(samples) + "\n"
    outlist = [str(cluster_time)[:5],str(init_time+centroid_time+cluster_time)[:5],str(sampleSize),str(samples)]
    outstring = ",".join(outlist)
    outstring = outstring + "\n"
    f.write(outstring)




# from daal.data_management import NumericTable
#
# NumericTable.