import math

def float_equal(f1, f2, delta=10**-6):
    return abs(f1 - f2) < delta

def centroid(coords):
    return [sum(x) / float(len(x)) for x in zip(*coords)]

def vec_add(vec1, vec2):
    return [x+y for x, y in zip(vec1, vec2)]

def vec_sub(vec1, vec2):
    return [x-y for x, y in zip(vec1, vec2)]

def vec_divide(vec, val):
    return [x / float(val) for x in vec]

def vec_multiply(vec1, vec2):
    return [x*y for x, y in zip(vec1, vec2)]

def normalize(vec):
    return vec_divide(vec, math.sqrt(sum([x*x for x in vec])))

def dot(vec1, vec2):
    return sum(vec_multiply(vec1, vec2))
