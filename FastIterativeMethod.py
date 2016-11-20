"""FastIterativeMethod.py"""
from pyspark import SparkContext
import numpy as np
import scipy.sparse as sps
import Utility
# Remove warning
import warnings
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 

from Preprocess import preprocess
from Query import query

class FastIterativeMethod:
  """ Fast Iterative Method class
  """

  _preprocessRan = False

  def __init__(self, edgeFilename, partitionFilename):
    """ Prepares the algorithm by loading all graph and node cluster partition information.

    Args:
      edgeFilename: (string) Edge filename string.
      partitionFilename: (strint) Node partition filename string.
    """
    i,j,value = np.loadtxt(edgeFilename).T
    i = i - 1
    j = j - 1
    self._n = max(np.append(i,j))
    i = np.append(i,self._n)
    j = np.append(j,self._n)
    # This way of extending the graph will need to be fixed. Corner value might be nonzero. Maybe? Self-looping? Probably right.
    value = np.append(value,0)
    out = sps.coo_matrix((value,(i,j)))
    self._n = max(out.shape)

    adj = out + out.T
    row_sum_inv = np.reciprocal(adj.sum(axis=0)).A1

    D_inv = np.diag(row_sum_inv)
    c = 0.85
    self._E = c*adj.dot(D_inv)

    self._nparts = 4
    self._partition_list = np.loadtxt(partitionFilename) + 1

  def getNodeCount(self):
    """ Get the number of nodes in this graph.

    Return:
      n: (int) Number of nodes.
    """
    return self._n

  def getReorderE(self):
    """ Gets the reordered graph of E.
    
    Return:
      reorder_E: (int) Reordered adjacency matrix E.
    """
    return self._reorder_E
        
  def preprocess(self):
    """ Preprocessing phase to execute boundary finding, reordering and equation calculation sequentially.
    """
    self._preprocessRan = True
    self._node_reorder2, self._reorder_E, self._L_inv, self._U_inv, self._L_k_inv, self._U_k_inv, self._boundary_start_number, self._index_start, self._T1, self._T2 = preprocess(self._partition_list, self._nparts, self._n, self._E)

  # TODO (cvwang): mapping from y to the reordered y needs to be done eventually
  def query(self, y):
    """ Equation to solve: (I - E) * x = y. Query phase.

    Args:
      y: (np.array) The query vector.
    
    Return:
      x: (np.array) The solved x vector.
    """
    if self._preprocessRan:
      return query(y, self._T1, self._T2, self._L_inv, self._U_inv, self._L_k_inv, self._U_k_inv, self._boundary_start_number, self._index_start, self._nparts, self._n)
    else:
      print 'Must run preprocess step before.'






