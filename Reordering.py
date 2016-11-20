"""Reordering.py"""
import numpy as np
import Utility

def reordering(partition_list, boundary_list, nparts, n, E):
  # node_reorder1 = np.argsort(partition_list)
  node_reorder1 = np.argsort(partition_list,kind='mergesort')
  B = np.sort(partition_list)
  index_start = np.nonzero(np.diff(np.append(0,B)))[0]
  index_end = np.append(index_start[np.arange(1,index_start.size)]-1, n-1)
  boundary_group = partition_list[boundary_list]
  boundary_reorder1 = np.argsort(boundary_group,kind='mergesort')
  B = np.sort(boundary_group)
  boundary_reorder_list = boundary_list[boundary_reorder1]
  E_boundary = E[np.ix_(boundary_reorder_list, boundary_reorder_list)]
  print E_boundary.sum().sum()
  bound_start = np.nonzero(np.diff(np.append(0,B)))[0]
  l = boundary_list.size
  bound_end = np.append(bound_start[np.arange(1,bound_start.size)]-1, l-1)

  v = np.zeros((l,1))
  for i in range(l): # This can be parallelized
    #TODO Be careful about the use of [0]. Make sure it makes sense.
    v[i] = np.where(node_reorder1 == boundary_list[i])[0]
  v = v.astype(int)
  node_reorder1[v] = 0
  C = node_reorder1[np.nonzero(node_reorder1)]
  B = partition_list[C]
  inner_start = np.nonzero(np.diff(np.append(0,B)))[0]
  inner_end = np.append(inner_start[np.arange(1,inner_start.size)]-1, C.size-1)
  inner_node_per_group = inner_end - inner_start + 1
  boundary_start_number = index_start + inner_node_per_group

  for i in range(nparts):
    E_boundary[np.ix_(range(bound_start[i],bound_end[i]+1),range(bound_start[i],bound_end[i]+1))] = 0

  print E_boundary.sum().sum()


  node_reorder2 = np.zeros(n)
  for i in range(nparts): # Can be done in parallel for each group. Concatenate the inner node list and boundary node list for each group.
    node_reorder2[range(index_start[i],boundary_start_number[i])] = C[range(inner_start[i],inner_end[i]+1)]
    node_reorder2[range(boundary_start_number[i],index_end[i]+1)] = boundary_reorder_list[range(bound_start[i],bound_end[i]+1)]
  node_reorder2 = node_reorder2.astype(int)
  reorder_E = E[np.ix_(node_reorder2,node_reorder2)];

  return (node_reorder2, E_boundary, reorder_E, index_start, boundary_start_number)
