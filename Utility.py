"""Utility.py"""
import numpy as np

def printVecFile(vec, filename):
  print 'Printing vector to %s' % filename
  vec.tofile(filename, sep='\n')

def printMatFile(mat, filename):
  print 'Printing matrix to %s' % filename
  with open(filename, 'w') as f:
    mat.tofile(filename, sep='\n')
    for i in range(mat.shape[0]):
      for j in range(mat.shape[1]):
        f.write('%e ' % mat[i][j])
      f.write('\n')
