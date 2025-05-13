#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
There follow the seven portions multiplying unity by two and
three to get the sequence 1, 2, 3, 4, 9, 8, 27. Having made this
division and related it to the sevenness of the planetary system,
Plato goes on to describe the filling in of the intervals. This is done
by placing two means between each of the powers of 2 and powers
of 3. These are the arithmetic and harmonic means which, with the
geometric mean, complete the triad of means. The means set up
proportional unions between extremes and are therefore in them­
selves the epitome, in mathematical terms, of the mediating
principle—in common with the definition of psyche ('soul') as
mediating between the metaphysical (intelligible) domain and the
physical (sensible) domain. The means themselves are set in a
hierarchical tendency, as one might call it. The geometrical is the
most heavenward (metaphysical), the harmonic the most central
(psychic and anthropological), and the arithmetic the most earth­
ward iDhvsical).

Figure 10. Tetraktys showing full display of musical ratios: the octave proportion of 2:1; the
musical fifth proportion of 3:2; the musical fourth proportion of 4:3.; and the tone interval of 9:8.
Figure 11. The arithmetic proportion with the three arithmetic 'means' of 3, 6 and 9.
"""

tr = [[1,3,9,27],[2,6,18],[4,12]]
dr = [[1,2,4,8 ],[3,6,12],[9,18]]
decad = [1,2,3,4,6,8,9,12,18,27];
art_means = [3,6,9]
harm_means = [
    [3,4,6],
    [6,8,12],
    [9,12,18]    
]

def ratio_seq(start,r,l = 6):
    ret = [start]
    for i in range(l-1):
     o = ret[i] * r
     ret.append(o)
    return ret

test = ratio_seq(1,2,4);
print(test)

class tetraktys:
    
    def __init__(self, decad_num):
        self.dr = ratio_seq(decad_num,2,4)
        self.tr = ratio_seq(decad_num,3,4)  
     


















