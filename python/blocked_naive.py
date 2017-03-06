import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import heapq
import time
import numpy as np

if len(sys.argv) != 3:
    print("args: InputDir K")
    sys.exit(0)

input_dir = sys.argv[1]
if input_dir[-1] == '/':
    input_dir[:-1]

k = int(sys.argv[2])

userweights_path = input_dir + "/s2_userWeights.csv"
itemweights_path = input_dir + "/s_itemWeights.csv"

user_matrix = np.loadtxt(userweights_path, dtype=np.float32, delimiter=",")
item_matrix =np.loadtxt(itemweights_path, dtype=np.float32, delimiter=",")

print("files loaded.")

t0 = time.time()
ratings = np.dot(user_matrix, item_matrix.T)
t1 = time.time()
print("gemm time: %f" % (t1-t0))

t2 = time.time()
for i in range(ratings.shape[0]):
    heap = []
    for j in range(k):
        heapq.heappush(heap, ratings[i][j])
    for j in range(k,ratings.shape[1]):
        if ratings[i][j] > heap[0]:
            heapq.heapreplace(heap, ratings[i][j])

t3 = time.time()

print("heap time: %f" % (t3-t2))

total_time = (t3-t2)+(t1-t0)
print("total time: %f" % total_time)