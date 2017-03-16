
from .. import tetrahedral
import unittest
import numpy as np

class TestIsoSurface(unittest.TestCase):

    def two_dots(self, x, y, z):
        if x==y==z==-8 or x==y==z==0:
            return 1
        return -1

    def test_isosurface_ep(self):
        f = self.two_dots
        mins = [-8] * 3
        maxes = [8] * 3
        deltas = [2] * 3
        eps = [[(-8, -8, -8), (-8, -8, 8)]]
        S = tetrahedral.TriangulatedIsosurfaces(mins, maxes, deltas, f, 0, eps)
        (points, triangles) = S.get_points_and_triangles()
        points = [tuple(int(i) for i in pt) for pt in points]
        #from pprint import pprint
        #print "points and triangles"
        #pprint(points)
        #pprint(triangles)
        triangle_vertices = set(frozenset(points[i] for i in triangle) for triangle in triangles)
        #print "triangle_vertices"
        #pprint(triangle_vertices)
        expected = set([frozenset([(-9, -9, -8), (-9, -8, -8), (-8, -8, -7)]),
                        frozenset([(-7, -8, -8), (-7, -8, -7), (-7, -7, -7)]),
                        frozenset([(-8, -8, -7), (-8, -7, -7), (-7, -7, -7)]),
                        frozenset([(-8, -8, -7), (-7, -8, -7), (-7, -7, -7)]),
                        frozenset([(-9, -9, -8), (-8, -9, -8), (-8, -8, -7)]),
                        frozenset([(-8, -7, -8), (-7, -7, -8), (-7, -7, -7)]),
                        frozenset([(-7, -8, -8), (-7, -7, -8), (-7, -7, -7)]),
                        frozenset([(-8, -7, -8), (-8, -7, -7), (-7, -7, -7)])])
        self.assertEqual(triangle_vertices, expected)
