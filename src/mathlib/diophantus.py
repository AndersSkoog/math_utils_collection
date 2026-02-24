# divide like diophantus
def divide_with_difference(whole, difference):
    """
    To divide a proposed number into two numbers having a given difference.
    Now, let the given number be 100, and the difference be 40 units.
    To find the numbers.
    Let the smaller be assigned to be 1 Number. Therefore, the greater will be 1
    Number, 40 units. The two together, therefore, become 2 Numbers, 40 units. But
    they were given to be 100 units. Thus, 100 units are equal to 2 Numbers, 40 units.
    And likes from likes From the 100 (units) I subtract 40 units, and from the
    2 Numbers and the 40 units, similarly, 40 units.  The remaining 2 Numbers are
    equal to 60 units; therefore, each one becomes 30 units.
    To the hypostases: the smaller will be 30 units, the greater 70 units, and the
    proof is obvious.
    """
    sml = (whole - difference) / 2
    grt = whole - sml
    return [grt, sml]

def divide_with_ratio(whole, ratio):
    """
    It is required to divide a proposed number into two numbers having a given ratio.
    Now, let it be proposed to divide 60 into two numbers having a triple ratio.
    Let the smaller be assigned to be 1 Number. Therefore, the greater will be 3 Numbers,
    so the greater is the triple of the smaller. Then the two must be equal to 60
    units. But the two, when added together, are 4 Numbers. Thus, 4 Numbers are equal to 60 units;
    therefore, the Number is 15 units. Thus, the smaller will be 15 units, and the greater 45 units.
    """
    sml = whole / (ratio + 1)
    grt = whole - sml
    return [grt, sml]

def divide_with_diff_and_ratio(whole, diff, ratio):
    """
    To divide a proposed number into two numbers having a given ratio and difference
    Now, let it be proposed to divide 80 into two numbers so that the greater is the
    triple of the smaller and, moreover, exceeds (it) by (an additional) 4 units.
    Let the smaller be assigned to be 1 Number. Therefore, the greater will be 3
    Numbers and 4 units; so the greater, being the triple of the smaller, exceeds more-
    over by (an additional) four units. Then, I want the two to be equal to 80 units. But
    the two, when added together, are 4 Numbers and 4 units. Thus, 4 Numbers and 4
    units are equal to 80 units.
    And I subtract likes from likes. Then, there remain 76 units equal to 4 Numbers,
    and the Number becomes 19 units
    """
    wsub = whole - diff
    sml = wsub / (ratio + 1)
    grt = (sml * ratio) + diff
    return [grt, sml]

def find_dividends(diff, ratio):
    """
    Now, let it be proposed that the greater is the quintuple of the smaller,
    and their difference makes 20 units.
    Let the smaller be assigned to be 1 Number. Therefore, the greater will be 5
    Numbers. Then, I want 5 Numbers to exceed 1 Number by 20 units. But their dif-
    ference is 4 Numbers. These are equal to 20 units.
    The smaller number will be 5 units, and the greater 25 units. And it is fulfilled,
    since the greater is the quintuple of the smaller, and the difference is 20 units.
    """
    part = diff / (ratio - 1)
    sml = part
    grt = part * ratio
    return [grt, sml]

def divide_by_ratios(whole, ratio1, ratio2):
    """
    Now, let it be proposed to divide 100 into two numbers such that the third of the
    first and the fifth of the second, if added together, make 30 units.
    I assign the fifth of the second to be 1 Number; (the second) itself, therefore,
    will be 5 Numbers. Then, the third of the first will be 30 units lacking 1 Number;
    (the first) itself, therefore, will be 90 units lacking 3 Numbers. Then, I want the
    two, if added together to make 100 units. But the two, when added together, make
    2 Numbers and 90 units. These are equal to 100 units.
    And likes from likes. Then, there remain 10 units equal to 2 Numbers;
    therefore, the Number will be 5 units.
    To the hypostases: I assigned the fifth of the second to be 1 Number;
    it will be 5 units; (the second) itself, therefore, will be 25 units.
    And (I assigned) the third of the first, to be 30 units lacking 1 Number; it will be | 25 units;
    (the first) itself, therefore, will be 75 units. And it is fulfilled,
    since the third of the first and the fifth of the second is 30 units,
    and these, when added together, make the proposed number
    """
    r1 = max(ratio1, ratio2)
    r2 = min(ratio1, ratio2)
    part1 = whole / (r1 + 1)
    part2 = part1 / r2
    grt = part1 * r1
    sml = part2 * r2
    return [grt, sml, grt + sml]