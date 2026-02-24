def gnomonic_number(n): return 1 if n == 1 else pow(n, 2) - pow(n - 1, 2)

def polygon_num(s, n):
    t1 = (n * (n - 1)) / 2
    t2 = s - 2
    res = t2 * t1 + n
    return 1 if n <= 1 else int(res)

def centered_polygon_num(s,n):
    return 1 if n <= 1 else int((s * pow(n,2) - s * n + 2) / 2)

def tetrahedral_numbers(n):
    return 1 if n <= 1 else int((n * ((n + 1) * (n + 2))) / 6)

def cubic_number(n):
    return 1 if n <= 1 else pow(n, 3)

def octahedral_number(n):
    return 1 if n <= 1 else int((n * (pow(2 * n, 2) + 1)) / 3)

def icosahedral_number(n):
     n2 = pow(n, 2)
     n5 = 5 * n
     return 1 if n <= 1 else int((n * (5*n2 - n5 + 2)) / 2)

def dodecahedral_number(n):
    nom = n * ((3*n) - 1) * ((3*n) - 2)
    return 1 if n <= 1 else int(nom / 2)

def stella_octangula_number(n):
    return 1 if n <= 1 else octahedral_number(n) + (8 * tetrahedral_numbers(n - 1))

def rhombic_dodecahedron_number(n):
    return 1 if n <= 1 else centered_cube_number(n) + (6 * m_gon_pyramid(4, n - 1))

def centered_cube_number(n):
    return 1 if n <= 1 else pow(n, 3) + pow(n - 1, 3)

def m_gon_pyramid(m, n):
    nom = n * (n + 1) * ((m - 2) * n - m + 5)
    dnom = 6
    return 1 if n <= 1 else int(nom / dnom)

def centered_mgon_pyramid(m, n):
    n2 = 2 * n - 1
    m_1 = m - 1
    nn = pow(n,2)
    nom = (n2 * m_1 * nn) - (m_1 * n + 6)
    return 1 if n <= 1 else int(nom/2)
    #return 1 if n <= 1 else int(((2*n - 1) * (s - 1) * pow(n,2) - (s - 1) * n + 6) / 2)

def seq(s_start,s_stop,init,fn):
    ret = init
    for i in range(s_start,s_stop):
        nv = fn(ret,i)
        ret.append(nv)
    return ret

def get_perfect(doubles_seq):
    s = sum(doubles_seq)
    last = doubles_seq[len(doubles_seq) - 1]
    return s * last

def polygon_diagonals(sides):
    return sides * (sides - 3) / 2

def algo_seq(n_start, n_stop, fn, init):
    ret = init
    for i in range(n_start, n_stop):
        v = fn(ret, i)
        if v:
         ret.append(v)
    return ret

def solid_numbers(n_iter):
    laterals_a = []
    laterals_b = []
    diagonals_a = []
    diagonals_b = []
    prev_s = 1
    prev_d = 1
    for i in range(1,n_iter):
      s = prev_s + prev_d
      d = prev_d + (prev_s * 2)
      s2 = s ** 2
      d2 = d ** 2
      prev_s = s
      prev_d = d
      laterals_a.append(s)
      laterals_b.append(s2)
      diagonals_a.append(d)
      diagonals_b.append(d2)
    return [laterals_a,laterals_b,diagonals_a,diagonals_b]