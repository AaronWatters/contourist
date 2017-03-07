from .. import triangulated
import unittest
import numpy as np

EXPECT_SVG = """
<svg height="300.0" width="300" viewBox="-1 -1 2 2">
<path stroke-width="0.02" stroke="black" fill="none" d="M0.00 0.00 L0.00 1.00 L1.00 1.00 Z" />
<path stroke-width="0.02" stroke="black" fill="none" d="M-1.00 -1.00 L-1.00 0.00" />
</svg>
"""

class TestMisc(unittest.TestCase):

    def test_svg(self):
        cseqs = [(True, [(0, 0), (0, 1), (1, 1)]),
                 (False, [(-1, -1), (-1, 0)])]
        svg = triangulated.contour_sequences_to_svg(cseqs)
        self.assertEqual(svg.strip(), EXPECT_SVG.strip())

    def test_adjacent_pairs(self):
        low_pair = (0,0)
        high_pair = (0,1)
        adj = list(triangulated.adjacent_pairs(low_pair, high_pair))
        expected = [((0, 0), (-1, 0)), ((0, 0), (1, 1)), ((1, 1), (0, 1)), ((-1, 0), (0, 1))]
        self.assertEqual(adj, expected)

class TestAdjacentPairs(unittest.TestCase):

    def test_vertical(self):
        L = set(triangulated.adjacent_pairs((0,0), (0,1)))
        expect = [((0, 0), (1, 1)), ((-1, 0), (0, 1)), ((1, 1), (0, 1)), ((0, 0), (-1, 0))]
        self.assertEqual(L, set(expect))

    def test_horizontal(self):
        L = set(triangulated.adjacent_pairs((0,0), (1,0)))
        expect = [((0, -1), (1, 0)), ((0, 0), (0, -1)), ((1, 1), (1, 0)), ((0, 0), (1, 1))]
        self.assertEqual(L, set(expect))

    def test_diagonal(self):
        L = set(triangulated.adjacent_pairs((0,0), (-1,-1)))
        expect = [((0, -1), (-1, -1)), ((-1, 0), (-1, -1)), ((0, 0), (0, -1)), ((0, 0), (-1, 0))]
        self.assertEqual(L, set(expect))

class TestDxDy(unittest.TestCase):

    def two_dots(self, x, y):
        if x==y==-4 or x==y==0:
            return 1
        return -1

    def clean_contours(self, contours):
        result = []
        for (closed, pts) in contours:
            tups = [(int(x*10), int(y*10)) for (x, y) in pts]
            result.append((closed, tups))
        return result

    def test_dxdy(self):
        f = self.two_dots
        C = triangulated.DxDy2DContour(-4, -4, 4, 4, 2, 2, f, 0)
        contours = C.get_contour_sequences()
        contours = self.clean_contours(contours)
        expected = self.clean_contours([
            (False, [(-4.0, -3.0), (-3.0, -3.0), (-3.0, -4.0)]),
            (True, [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, -1.0), (-1.0, 0.0)])
        ])
        self.assertEqual(contours, expected)

    def test_dxdy_endpoint(self):
        #print "start dxdy_endpoint"
        f = self.two_dots
        ep = [[(-4,-4), (-4, -1)]]
        C = triangulated.DxDy2DContour(-4, -4, 4, 4, 1, 1, f, 0, ep)
        contours = C.get_contour_sequences()
        contours = self.clean_contours(contours)
        expected = self.clean_contours([(False, [(-4.0, -3.5), (-3.5, -3.5), (-3.5, -4.0)])])
        self.assertEqual(contours, expected)

class TestGrid2DContour(unittest.TestCase):

    def test_line(self):
        def linear(x,y):
            return x + y 
        endpoints = [[(0,0), (2,2)]]
        G = triangulated.Grid2DContour(2, 2, linear, 1.5, endpoints)
        S = G.get_contour_sequences()
        [(closed, contour)] = S 
        self.assertFalse(closed)
        expected = np.array([(1.0, 0.5), (0.75, 0.75), (0.5, 1.0)])
        #print "contour", contour
        assert np.allclose(expected, contour)

    def test_dot(self):
        def dot(x,y):
            if x == 1 and y == 1:
                return 2
            return 0
        endpoints = [[(0,0), (1,1)]]
        G = triangulated.Grid2DContour(3, 3, dot, 1, endpoints)
        S = G.get_contour_sequences()
        [(closed, contour)] = S 
        self.assertTrue(closed)
        expected = [[0.5, 0.5], [1.0, 0.5], [1.5, 1.0], [1.5, 1.5], [1.0, 1.5], [0.5, 1.0]]
        #print "contour", closed, contour.tolist()
        # xxx it may be possible for the contour to reverse???
        assert np.allclose(np.array(expected), contour)
