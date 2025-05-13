from itertools import product

def num_conj(arr):
  ret_str = ''
  for v in arr:
      ret_str += v
  return int(ret_str)


def generate_tensor_indices(rank, N):
    """
    Generates all index combinations for a tensor of a given rank and dimension size N.
    The index combinations will be in the form of tuples (i1, i2, ..., irank).
    """
    # Create a range for each dimension (each index runs from 1 to N)
    index_ranges = [range(1, N + 1)] * rank

    # Use itertools.product to generate the Cartesian product of these ranges
    return list(product(*index_ranges))

def tensor_el(rank,stack,row,pos):
    return [stack]


def tensor_index(rank,stack,row,pos):
  first = num_conj(['1'] * rank)
  s = (1000 * stack) if stack > 1 else 0
  r = (100 * row) if row > 1 else 0
  p =
  return first + (1000 * stack) + (100 * row) + (pos % 9)

print(tensor_index(4,1,3,2))





"""
11, 12
21, 22

111,112,113    113
211,221,231    231
311,321,331            311,321,3
"""
 # Generates all index combinations

"""
[(0),(1),(2)  [3**2 + 1,3**2 + 2,3**2 + 3, 
 (3),(4),(5)   3**2 + 1,3**2 + 2,3**2 + 3,
 (6),(7),(8)], 3**2 + 1,3**2 + 2,3**2 + 3]

def generate_tensor_indices(N):
    indices = []
    for i in range(1, N + 1):  # Stack index
        for j in range(1, N + 1):  # Row index
            for k in range(1, N + 1):  # Column index
                for l in range(1, N + 1):  # Position in column
                    indices.append((i, j, k, l))  # Store the index tuple
    return indices

generate_tensor_indices()


def tensor_order(rank,stack,row):
  tup = tuple([1] * rank)
  tup[0] = stack
  #1111,1112,1113,1114
  #2111,2112,2113,2114
  #3111,3112,3113,3114

def tensor_row(rank,stack,i):
  #1111,1112,1113 | 2121,2132,2133 | 3111,3112,3113,3114
  rr = [stack,]
  rr[0] = stack
  indices = list(product(range(1,rank), repeat=rank))
  print(indices)
  return indices

tensor_row(4,0,0)

def tensor(rank):
 stacks = 1 if rank < 3 else rank
 first_row = [[1] for i in range (rank)]
 #ind_row = lambda s,i: inds[((3**1) + 1) + i] if s % 2 == 0 else inds[(3*s) + i]
 mtx = [[ind_row(s,0),ind_row(s,1),ind_row(s,2)] for s in range(1,stacks)]
 return mtx
"""







