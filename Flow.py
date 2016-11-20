"""Flow.py"""
import numpy as np
# import scipy as sc
from scipy import linalg
import Utility

def flow(E_boundary, reorder_E, index_start, boundary_start_number, n, nparts):
  
  # Preprocessing
  reorder_W = np.eye(n) - reorder_E
  #TODO Should I just pass this from reordering.py?
  index_end = np.append(index_start[np.arange(1,index_start.size)]-1, n-1)
  H3 = [None]*nparts
  H4 = [None]*nparts
  L = [None]*nparts
  U = [None]*nparts
  wi = [None]*nparts

  for i in range(nparts):
    wi[i] = reorder_W[np.ix_(range(index_start[i],index_end[i]+1),range(index_start[i],index_end[i]+1))]
    _, L[i], U[i] = linalg.lu(wi[i])
    #TODO see if these inverses can be sped up with some specialed triangular matrix inverse
    L[i] = np.linalg.inv(L[i])
    U[i] = np.linalg.inv(U[i])

    # You may not need some +1 here
    H3[i] = np.dot(U[i][np.ix_(range(boundary_start_number[i]-index_start[i],U[i].shape[0]),range(U[i].shape[1]))], L[i][np.ix_(range(L[i].shape[0]),range(boundary_start_number[i]-index_start[i]))])
    H4[i] = np.dot(U[i][np.ix_(range(boundary_start_number[i]-index_start[i],U[i].shape[0]),range(U[i].shape[1]))], L[i][np.ix_(range(L[i].shape[0]),range(boundary_start_number[i]-index_start[i],L[i].shape[1]))])

  L_inv = linalg.block_diag(*L)
  U_inv = linalg.block_diag(*U)
  H_3 = linalg.block_diag(*H3)
  H_4 = linalg.block_diag(*H4)

  # Main equation
  I = np.eye(E_boundary.shape[0])
  T1 = np.dot(E_boundary,H_4)
  T2 = np.dot(E_boundary,H_3)
  T =  I - T1
  _, L1, U1 = linalg.lu(T)
  L_k_inv = np.linalg.inv(L1)
  U_k_inv = np.linalg.inv(U1)

  return (L_inv, U_inv, L_k_inv, U_k_inv, T1, T2)
