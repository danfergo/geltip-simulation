import math


def distance(p1, p2):
    return math.sqrt(sum([math.pow(p1[i] - p2[i], 2) for i in range(len(p1))]))
