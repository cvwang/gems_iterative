"""FastIterativeMethodDriver.py"""

import numpy as np
from FastIterativeMethod import FastIterativeMethod

edgeFilename = 'out.petster-friendships-hamster-uniq'
partitionFilename = 'friendship.graph.part.4'
fastIterativeMethod = FastIterativeMethod(edgeFilename, partitionFilename)
fastIterativeMethod.preprocess()

c = 0.85
n = fastIterativeMethod.getNodeCount()
y = (1-c) * np.ones((n,1)) / n;

x = fastIterativeMethod.query(y)
reorder_E = fastIterativeMethod.getReorderE()

I = np.eye(reorder_E.shape[0])
Standard = np.dot(np.linalg.inv((I - reorder_E)), y)
# Utility.printVecFile(Standard, 'output.txt')

print 'Diff Score: Python x vs. Python inverse x:'
""" Max Score """
# diffScore = np.amax(x - Standard) / np.amax(Standard)
""" Norm Score """
diffScore = np.linalg.norm(x - Standard) / np.linalg.norm(Standard)
print diffScore


