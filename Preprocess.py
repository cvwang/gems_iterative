"""Preprocess.py"""
from FindBoundary import find_boundary
from Reordering import reordering
from Flow import flow
import Utility

def preprocess(partition_list, nparts, n, E):

  boundary_list = find_boundary(E, partition_list, n)
  node_reorder2, E_boundary, reorder_E, index_start, boundary_start_number = reordering(partition_list, boundary_list, nparts, n, E)
  L_inv, U_inv, L_k_inv, U_k_inv, T1, T2 = flow(E_boundary, reorder_E, index_start, boundary_start_number, n, nparts)

  return (node_reorder2, reorder_E, L_inv, U_inv, L_k_inv, U_k_inv, boundary_start_number, index_start, T1, T2)