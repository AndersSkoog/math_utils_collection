"""
Instead of thinking of rational numbers a/b as simply dividing a into b parts,
We treat it as defining a recursive field based on the integer b (the "host" integer),
The numerator a defines how many iterations we apply of proportional recursion (or subdivision),
This results in:
A minimal unit (the deepest recursive subdivision),
A maximum value (number of additions of the minimal unit).
This reframes a/b as: a recursive steps into the field of reciprocals of integer b.

the proportianal number system recognize that we should not treat rational numbers to be anything else than an expression
of a reciprocials of an integer b where a/b = 1/b * a
we get the minimal unit u from dividing the reciprocal 1/b with it self a number of times,
"""

class ProportionalNumber:
    def __init__(self, a, b):
        self.a = a      # numerator
        self.b = b    # denominator (base integer field)
        self.reci = 1 / b # base reciprocal: 1/b
        self.rational = a / b
        self.units_in_reci = pow(b,a)
        self.units_in_rational = self.units_in_reci * a
        self.units_in_maxval = self.units_in_rational * pow(b,a)
        self.unit = self.reci / self.units_in_reci
        self.max_val = self.unit * self.units_in_maxval
        self.field = [self.unit * i for i in range(1,self.units_in_maxval)]
        print(self.field)


test = ProportionalNumber(3,5)
