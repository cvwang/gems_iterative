"""Flow.py"""
import numpy as np

def query(y, T1, T2, L_inv, U_inv, L_k_inv, U_k_inv, boundary_start_number, index_start, nparts, n):

  yb = [None]*nparts
  yi = [None]*nparts
  keysub = [None]*nparts
  index_end = np.append((index_start[1:] - 1), n-1)
  key = np.dot(U_inv, L_inv) 
  for i in range(nparts): # Parallelize
    yb[i] = y[boundary_start_number[i]:index_end[i]+1]
    yi[i] = y[index_start[i]:boundary_start_number[i]]
    keysub[i] = key[np.ix_(range(key.shape[0]),range(boundary_start_number[i],index_end[i]+1))]
  y_b = np.concatenate(yb)
  y_i = np.concatenate(yi)
  b = np.dot(T1,y_b) + np.dot(T2,y_i)
  f = np.dot(U_k_inv, np.dot(L_k_inv, b))
  key_sub = np.concatenate(keysub,1)
  e = np.dot(key_sub, f)
  x = np.dot(key, y) + e
  
  return x