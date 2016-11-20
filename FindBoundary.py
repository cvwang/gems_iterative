"""FindBoundary.py"""
import numpy as np

def find_boundary(E, partition_list, n):
  boundary_list = np.zeros(n)
  row, col = np.nonzero(E)
  indicator = partition_list[row] - partition_list[col]
  b_ind_row =  np.nonzero(indicator)
  boundary1 = row[b_ind_row]
  boundary2 = col[b_ind_row]
  boundary_list[np.append(boundary1, boundary2)] = 1
  boundary_ind = np.nonzero(boundary_list)
  return boundary_ind[0]