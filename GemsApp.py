"""GemsApp.py"""
from pyspark import SparkContext
import numpy as np
import scipy.sparse as sps
import Utility

from Preprocess import preprocess
from Query import query

# Remove warning
import warnings
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 

# # Spark code

# logFile = '/Users/cvwang/spark-2.0.1-bin-hadoop2.7/README.md'  # Should be some file on your system
# sc = SparkContext('local', 'GEMS App')
# logData = sc.textFile(logFile).cache()

# numAs = logData.filter(lambda s: 'a' in s).count()
# numBs = logData.filter(lambda s: 'b' in s).count()
# print('Lines with a: %i, lines with b: %i' % (numAs, numBs))

# numCs = logData.filter(lambda s: 'cluster' in s).count()
# print('Lines with cluster: %i' % (numCs))

# # Parallelized Collections
# data = [1, 2, 3, 4, 5]
# distData = sc.parallelize(data)
# sumData = distData.reduce(lambda a, b: a + b)
# print 'sumData: %d' % sumData

# # External Data
# distFile = sc.textFile('hdfs://localhost:9000/user/cvwang/input')
# sumLines = distFile.map(lambda s: len(s)).reduce(lambda a, b: a + b)
# print 'sumLines: %d' % sumLines


# GEMS Lab code

i,j,value = np.loadtxt('out.petster-friendships-hamster-uniq').T
i = i - 1
j = j - 1
n = max(np.append(i,j))
i = np.append(i,n)
j = np.append(j,n)
value = np.append(value,0)
out = sps.coo_matrix((value,(i,j)))
n = max(out.shape)

adj = out + out.T
row_sum_inv = np.reciprocal(adj.sum(axis=0)).A1

D_inv = np.diag(row_sum_inv)
c = 0.85
E = c*adj.dot(D_inv)

nparts = 4
partition_list = np.loadtxt('friendship.graph.part.4') + 1

node_reorder2, reorder_E, L_inv, U_inv, L_k_inv, U_k_inv, boundary_start_number, index_start, T1, T2 = preprocess(partition_list, nparts, n, E)

y = (1-c) * np.ones((n,1)) / n;
x = query(y, T1, T2, L_inv, U_inv, L_k_inv, U_k_inv, boundary_start_number, index_start, nparts, n)

I = np.eye(reorder_E.shape[0])
Standard = np.dot(np.linalg.inv((I - reorder_E)), y)
# Utility.printVecFile(Standard, 'output.txt')

print 'Diff Score: Python x vs. Python inverse x:'
""" Max Score """
# diffScore = np.amax(x - Standard) / np.amax(Standard)
""" Norm Score """
diffScore = np.linalg.norm(x - Standard) / np.linalg.norm(Standard)
print diffScore
