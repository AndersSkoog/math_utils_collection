from itertools import combinations
from num import catalan

# get the number of triangulations of a regular n-gon
def tri_count(n:int):return catalan(n-2)

#get all diagonals of a regular n-gon
def diagonals(n:int):return [(i, j) for i in range(1, n) for j in range(i + 2, n + 1) if not (i == 1 and j == n)]

def do_intersect(d1, d2):
    #Check if two diagonals intersect.
    (a, b), (c, d) = sorted(d1), sorted(d2)
    return (a < c < b < d) or (c < a < d < b)

def is_valid_triangulation(d):
    #Check if the set of diagonals forms a valid triangulation.
    for d1, d2 in combinations(d, 2):
        if do_intersect(d1, d2):
            #print(d1,d2)
            return False
    return True


def generate_triangulations(n):
    #Generate all sets of diagonals that can form a non-intersecting triangulation in an n-gon
    diags = diagonals(n)
    triangulations = []
    """
    The Triangulation Diagonal Count Relation is a mathematical expression that quantifies the number of diagonals 
    required to triangulate a regular n-gon. 
    It states that the number of non-intersecting diagonals needed to divide a regular n-gon into triangles is given by n - 3
    """
    triangulation_diagonal_count = n - 3
    for diag_comb in combinations(diags, triangulation_diagonal_count):
        if is_valid_triangulation(diag_comb):
            triangulations.append(diag_comb)

    return triangulations

def generate_frieze_pattern(n,I):
 #n = integer representing number of vertices of an n-gon, I = index for picking a triangulation
 id_order = list(range(1,n + 1))
 tri_index = int(I % tri_count(n))
 tri_diagonals = generate_triangulations(n)[tri_index]
 flat_list = [item for sublist in tri_diagonals for item in sublist]
 frieze_seq = []
 for v in id_order:
   if v in flat_list:
       frieze_seq.append(flat_list.count(v) + 1)
   else:
       frieze_seq.append(1)
 r_1 = [1] * (n + 4)
 r_2 = frieze_seq + frieze_seq[0:4]
 r_3 = [int(((r_2[i-1] * r_2[i]) - 1) / r_1[i-1]) for i in range(1, n + 4)]
 r_4 = [int(((r_3[i-1] * r_3[i]) - 1) / r_2[i]) for i in range(1, n + 3)]
 r_5 = [int(((r_4[i-1] * r_4[i]) - 1) / r_3[i]) for i in range(1, n + 2)]
 r_6 = [int(((r_5[i-1] * r_5[i]) - 1) / r_4[i]) for i in range(1, n + 1)]
 return [r_1,r_2,r_3,r_4,r_5,r_6]
