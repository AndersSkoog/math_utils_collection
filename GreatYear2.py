from mpmath.libmp import agm_fixed

import math_utils
import math

from Diophantus import divide_with_ratio

"""
Each is in a certain sense a Great Year: the sum of the two we may call the
Greatest Year—the cycle which sees the Universe begun, completed, and ended. E
Each is one ‘journey round’ or zrepiodos,
and it is therefore with perfect justice that Plato uses this word
to denote the time which it takes to fashion the Divine Creature.
The srepiodos of the Oeiov yevyntév is thus identical
with the Great Year. Let us now see how Plato defines its
duration in days. Plato builds up the number of the Macro-
cosm from the number which expresses the gestation of the
Microcosm, Man.
Taking the sum of the 3 numbers, 3, 4, 5,
the sides of the Pythagorean triangle, he multiplies them by
the hypotenuse 5, and raises the product to the power denoted
by 4, which is the number of the remaining side of the triangle.
The sepiodos of the Oetov yevvnréy is therefore produced by


Thus among the Indians 
360 years was ‘a year of the gods,’ 
3600 a ‘cycle of Brihaspati,’ 
216000 a ‘cycle of Prajapati, 
4,320,000 an ‘age of the gods,’ 
and the ‘kalpa’ 1000 ‘ages of the gods’ or one ‘day of Brahma, 
while twice this number, or 8,640,000,000 years was ‘a day and a night of Brahma’

2700 : 3600 :: 3600 ; 4800 ie. 9: 12::12:16.
But we have not yet sounded the full depth of meaning in the word yewpetpixos. 
It is the climax of the whole—a climax not unworthy of the Muses.
The whole number, 25,920,000 days or 72000 years, is the measure of the Earth—from the day of her
begetting till the day she issued from the womb, and from the
hour of her birth till her soul returns to God who gave it.
For 36000 years she waxes: through 36000 years she wanes’.
We have now to explain the words rosovrou Kvpios, apes-
VOVOV TE Kal YELpovwVv yeverewn.





(1) w^3 + x^3 + y^3 = z = 216,
where w = 3, x = 4, y = 5.
i,e the pythagorean triangle.
(2) [(w+a+y) x 5]^4 = 3600^2 = 4800 * 2700.

At the time of conception
the sun must be in some one sign, and in some part of it: the
name of this part is the locus conceptioms: they are in all 360,
30 in each sign: and they are called by the Greeks poipaz, eo
videlicet quod deas fatales nuncupant Moeras et eae particulae
nobis velut. fata sunt’.
"""
#signs = ["A","B","C","D","E","F","G","B","H","K","L","M","N"]
#year = [30 * i for i in range(0,12)]
#year_signs = [[signs[i],60 * i] for i in range(0,12)]
#great_year = 360 * 36000
#full_great_year = 360 * 72000

"""
begining of new golden age, 
    marks the end of an iron age 
    marks the start of a new bronze age towards an iron age
    marks the start of retrogression of previous golden and silver ages.


Example framework for eternal return
imagine a hexagon, and 12 instances of self grouped in pairs, 
where each pair is assoicated with a start vertex




"""



def findIrrationals(n):
  res = []
  for i in range(1,n):
      for j in range(1,n):
          v = ((i / j) ** 2) != 2
          res.append(f"{i}/{j}")
  return res








def perfects(limit):
  ret = []
  for i in range(2,limit):
    v = math.sqrt(i)
    iv = int(v)
    rv = v - iv
    if(rv == 0):
      ret.append({"power":i,"root":iv,"cubling":iv*pow(iv,2)})
  return ret

cubling         = lambda v: v * pow(v,2)
pyth_area_tri   = lambda n: int(((3*n) * (4*n)) / 2)
pyth_area_rect  = lambda n: (3*n) * (4*n)
pyth_hypot      = lambda n: int(math.hypot(3*n,4*n))
pyth_prod       = lambda n: math.prod([3*n,4*n,pyth_hypot(n)])
pyth_tripple    = lambda n: [3*n,4*n,int(math.hypot(3*n,4*n))]
pyth_trip_pow   = lambda n: [pow(v,2) for v in pyth_tripple(1)]
pyth_side_a     = lambda n: 3*n
pyth_side_b     = lambda n: 4*n
prop            = lambda w,a,b: w / (a/b)
frac_series     = lambda w,q,n: [int(w * (1/q)) * i for i in range(1,n+1)]
pyth_tri = [3,4,5]
cyc_unit = 360
philo_years = 100
unit_area = pyth_area_tri(1) # 6
nup_a = pyth_side_a(unit_area) # 3*6=18
nup_b = pyth_side_b(unit_area) # 4*6=24
nup_num = pyth_area_tri(unit_area) #(18*24) / 2 = 216
nup_hyp = pyth_hypot(unit_area) # math.hypot(18,24) = 30
nup_ratio_a = nup_a / nup_b # 3/4
nup_ratio_b = nup_b / nup_a # 4/3
cyc_ratio_a = int(cyc_unit * nup_ratio_a) # 270
cyc_ratio_b = int(cyc_unit * nup_ratio_b) # 480
greatyear_days = (cyc_ratio_a * cyc_ratio_b) * philo_years # (270*480)*100 = 12960000
greatyear_years = greatyear_days // 360 # 36000
cosmicyear_days = greatyear_days * 2
cosmicyear_years = greatyear_years * 2
cosmic_harm = cosmicyear_years / nup_num # 333.333333333
ages_of_man = 4
age_days = greatyear_days / ages_of_man
birth_months = [4+3,4+4,4+5] # [7,8,9]
gestation_periods = [30 * m for m in birth_months] # [210,240,270]
gestation_periods_plato = [v+6 for v in gestation_periods]
harms = {"forth":(4/3),"fifth":(3/2),"octave":(12/6)}

def cosmic_date(n):
 years = n // 360
 day_in_year = n % (360 * years)
 month_in_year = day_in_year // 12
 day_in_month = day_in_year % 30
 ret = {
   "years":years,
    "month_in_year":month_in_year,
    "day_in_year":day_in_year,
    "day_in_month":day_in_month
 }
 return ret

cons_vals = [
  divide_with_ratio(100*360,1),
  divide_with_ratio(100*360,2),
  divide_with_ratio(100*360,3),
  divide_with_ratio(100*360,4)
]

possible_births = ""


def philo(b):
  births = [30*7,30*8,30*9]
  death = ""

  life_span = (100 * 360)
  cons = (life_span * (1/3))

  child_birth = cons + (30 * 9)
  child_cons = child_birth + ((36 * 360) + (30 * 3))
  new_birth = child_cons + (30 * 9)
  return [life_span,cons,child_birth,child_cons,new_birth]


  #child_birth = cons + 270
  #child_cons  = child_birth + cons
  #death = child_cons + 270
  #new_birth = b + death
  #return {new_birth:"new_birth","cons":cons_1,"child_birth":child_birth,"child_cons":child_cons,"death":death}

print(philo(0))
#cons_day = [90 + (life_span * n) for n in range(0,101)]
#birth_day = [v + 270 for v in cons_day]
#death_day = [v + life_span for v in birth_day]
#print("consdays:",cons_day)
#print("birthdays:",birth_day)
#print("deathdays:",death_day)
#cons_day = 0
#birth_day = gestation # 360
#death_day = birth_day + (100 * 360)
#philo_life = [cons_day,gestation,birth_day,death_day]
#print(philo_life)


obj = dict(
  unit_cycle=360,
  years_of_philosopher=100,
  ages_of_man=4,
  pyth_tripple=[3,4,5],
  days_greatyear=(3*4*5) ** 4,
  nuptial_tripple=[18,24,30],
  nuptial_number=pyth_area_tri(6),
  nuptial_hypot=pyth_hypot(6),
  nuptial_ratio=pyth_area_tri(6) * pyth_hypot(6),
  cosmic_a=((pyth_area_tri(6) * pyth_hypot(6)) * 100) / 18,
  cosmic_b=((pyth_area_tri(6) * pyth_hypot(6)) * 100) / 24
)
#print(c_tri)
#print(120-90,150-120)
#print((pyth_area_tri(30) / 2) * 2)
#print(((216 * 30) * 100) / 18)
#print(((216 * 30) * 100) / 24)
#print(pyth_side_a(30))
#print(pyth_side_b(30))
#print(pyth_hypot(30))
#print(pyth_hypot(pyth_area(2)))



#print(nuptial_hypot)
#print(nuptial_area * nuptial_hypot)
#print(cosmic_ratio)

#fac_60      = utils.factors(60)
#first_sqr   = first_area ** 2 # 6^2 = 36
#first_cube  = first_area ** 3 # 6^3 = 216 is the first cube of the first area
#days_great_year  = (3*4*5) ** ages_of_man #prod of pythagorean tripple is 60^4=12960000
#days_cosmic_year = days_great_year * 2
#years_great_year = days_great_year // unit_cycle
#years_cosmic_year = years_great_year * 2
#cosmic_ratio = days_cosmic_year / first_cube

#print(pyth_area(60))

#powers_of_3 = [3*n for i in range(1,12)][-1]
#powers_of_4 = [pow(3,i) for i in range(1,12)][-1]
#print(pyth_areas)
#print(powers_of_3)

#great_area = 2700 * 4800
#great_hyp = math.hypot(2700,4800)



#ages_of_man = 4
#days_greatyear  = first_prod**ages_of_man // unit_cycle # 36000
#years_greatyear = days_greatyear // 360
#days_universe   = days_greatyear * 2
#years_universe  = years_greatyear * 2
#print(days_universe / 216)
#print(root_by_sqr)
#print(pyth_hypots)
#print(pyth_areas)










"""
great_area = 2700 * 4800
great_hyp = math.hypot(2700,4800)
root_and_sqr = [v*v**2 for v in pyth_tri]
print(root_and_sqr)

#(math.hypot(2700,4800))
great_year = great_area / unit_cycle # = 36000
areas = [12 * (n ** 2) for n in range(1, 901)][-1]
print(areas)
print(great_hyp / 360)
print(6*36)



def pyth_tri_area_pow(n):
  a = 3**n
  b = 4**n
  c = int(math.hypot(a,b))
  return int((a*b)/2)

def pyth_tri_area(n):
  a = 3*n
  b = 4*n
  c = int(math.hypot(a, b))
  area = int((a * b) / 2)
  return area

def point_on_circle(y,r):
    return math.sqrt(r - pow(y, 2))

def pyth_rect(n):
  if isinstance(n,int):
    a = 3*n
    b = 4*n
    c = int(math.hypot(a,b))
    area = int(a * b)
    return dict(A=a, B=b, Hyp=c, Area=area)

pyth_area_pow_series = [pyth_tri_area_pow(i) for i in range(1,12)]
pyth_area_series = [pyth_tri_area(i) for i in range(1,12)]
pyth_area_prod_series = [p[0] * p[1] for p in utils.adj(pyth_area_series)]
pyth_area_prod_pow_series = [p[0] * p[1] for p in utils.adj(pyth_area_pow_series)]
#print(utils.adj(pyth_area_series))
print(pyth_area_series)
print(pyth_area_pow_series)
print(pyth_area_prod_series)
print(pyth_area_prod_pow_series)
print([36000 / pyth_area_prod_series[i] for i in range(0,len(pyth_area_prod_series))])
pyth_tri_areas = [6, 24, 54, 96, 150, 216, 294, 384, 486, 600, 726]
pyth_tri_pow_areas = [6, 72, 864, 10368, 124416, 1492992, 17915904, 214990848, 2579890176, 30958682112, 371504185344]
pyth_tri_areas_adj_prod = [144, 1296, 5184, 14400, 32400, 63504, 112896, 186624, 291600, 435600]
factors_of_360 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30, 36, 40, 45, 60, 72, 90, 120, 180]
#print(utils.factors(360))
#pyth_triangles = [pyth_triangle(i) for i in range(1,36)]
#print([t.get("A") * t.get("B") * t.get("Hyp") for t in pyth_triangles])
#print([t.get("A") * t.get("B") for t in pyth_triangles])
#print([pyth_triangle(i) for i in range(1,12)])
"""


