#
# Ported from http://crcv.ucf.edu/source/3D.
# Original authors: Paul Scovanner, Saad Ali, and Mubarak Shah
#           A 3-Dimensional SIFT Descriptor and its Application to Action Recognition
#           ACM MM 2007
#

import math
from utils import *

def sphere_tri(radius=1, maxlevel=0, cache={}):
    if (radius, maxlevel) in cache:
        return cache[(radius, maxlevel)]

    t = (1.0 + math.sqrt(5.0)) / 2.0
    tau = t / math.sqrt(1 + t**2)
    one = 1.0 / math.sqrt(1 + tau**2)

    vertices = {
         1: [  tau,  one,    0 ],
         2: [ -tau,  one,    0 ],
         3: [ -tau, -one,    0 ],
         4: [  tau, -one,    0 ],
         5: [  one,   0 ,  tau ],
         6: [  one,   0 , -tau ],
         7: [ -one,   0 , -tau ],
         8: [ -one,   0 ,  tau ],
         9: [   0 ,  tau,  one ],
        10: [   0 , -tau,  one ],
        11: [   0 , -tau, -one ],
        12: [   0 ,  tau, -one ],
    }
   
    faces = [  [5,  8,  9],
               [5, 10,  8],
               [6, 12,  7],
               [6,  7, 11],
               [1,  4,  5],
               [1,  6,  4],
               [3,  2,  8],
               [3,  7,  2],
               [9, 12,  1],
               [9,  2, 12],
              [10,  4, 11],
              [10, 11,  3],
               [9,  1,  5],
              [12,  6,  1],
               [5,  4, 10],
               [6, 11,  4],
               [8,  2,  9],
               [7, 12,  2],
               [8, 10,  3],
               [7,  3, 11], ]

	
    for level in range(1, maxlevel+1):
        print "Refine: iteration #",level
        faces, vertices = mesh_refine(faces, vertices)
        vertices = sphere_project(vertices, radius)

    centers = {}
    for i in range(len(faces)):
        center = centroid([vertices[x] for x in faces[i]])
        centers[i] = normalize(center)
    
    cache[(radius, maxlevel)] = faces, vertices, centers
    return faces, vertices, centers

def mesh_refine(faces, vertices):
    num_faces = len(faces)
    f2 = [[0]*3 for _ in range(num_faces*4)]

    for f in range(num_faces):
        NA = faces[f][0]
        NB = faces[f][1]
        NC = faces[f][2]

        A = vertices[NA]
        B = vertices[NB]
        C = vertices[NC]

        a = vec_divide(vec_add(A, B), 2.0)
        b = vec_divide(vec_add(B, C), 2.0)
        c = vec_divide(vec_add(C, A), 2.0)

        Na = find_vertex(vertices, a)
        Nb = find_vertex(vertices, b)
        Nc = find_vertex(vertices, c)

        f2[(f+1)*4 - 3 - 1] = [NA, Na, Nc]
        f2[(f+1)*4 - 2 - 1] = [Na, NB, Nb]
        f2[(f+1)*4 - 1 - 1] = [Nc, Nb, NC]
        f2[(f+1)*4 - 0 - 1] = [Na, Nb, Nc]

    return f2, vertices

def find_vertex(vertices, vertex):
    max_index = -1
    for vertex_index, coord in vertices.items():
        max_index = max(max_index, vertex_index)
        if float_equal(coord[0], vertex[0]) and \
                float_equal(coord[1], vertex[1]) and \
                float_equal(coord[2], vertex[2]):
            return vertex_index
    vertices[max_index + 1] = vertex
    return max_index + 1

def sphere_project(vertices, radius=1.0, centroid=(0,0,0)):
    xo, yo, zo = centroid
    res = {}
    for face, (X, Y, Z) in vertices.items():
        theta = math.atan2((Y-yo), (X-xo))
        phi = math.atan2(math.sqrt((X-xo)**2 + (Y-yo)**2), Z - zo)

        res[face] = [radius * math.sin(phi) * math.cos(theta),
            radius * math.sin(phi) * math.sin(theta),
            radius * math.cos(phi)]

    return res

if __name__ == '__main__':
    print sphere_tri()
