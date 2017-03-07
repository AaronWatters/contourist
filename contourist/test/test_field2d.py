from .. import field2d
import unittest
import numpy as np

class TestGrid0(unittest.TestCase):

    materialize = False
    cache = False

    def test_f2dgrid(self):
        xmin = -10
        ymin = -20
        xmax = 30
        ymax = 50
        dx = 10.0
        dy = 20.0
        def function(x,y):
            return (x + 100) * 1000 + (y + 100)
        grid = field2d.Function2DGrid(xmin, ymin, xmax, ymax, dx, dy, function, self.materialize, self.cache)
        self.grid = grid
        for iteration in (1,2):
            assert np.allclose(grid.to_grid_coordinates((-10, -20)), (0,0))
            assert np.allclose(grid.from_grid_coordinates((0,0)), (-10, -20))
            assert np.allclose(grid.to_grid_coordinates((0,0)), (1,1))
            assert np.allclose(grid.from_grid_coordinates((1,1)), (0,0))
            assert np.allclose(grid.grid_function(0, 0), 90080)
            assert np.allclose(grid.grid_function(4, 3), 130140)
            S = set(tuple(x) for x in grid.surrounding_vertices((5,5)))
            self.assertEqual(set([(1, 2), (1, 1), (2, 1), (2, 2)]), S)
        self.afterwards()

    def afterwards(self):
        grid = self.grid
        assert grid.materialized_array is None
        assert len(grid.cache) == 0

class TestGrid1(TestGrid0):

    cache = True

    def afterwards(self):
        grid = self.grid
        assert grid.materialized_array is None
        self.assertEqual({(0, 0): 90080.0, (4, 3): 130140.0}, grid.cache)


class TestGrid2(TestGrid0):

    materialize = True

    def afterwards(self):
        grid = self.grid
        self.assertEqual(grid.cache, {})
        #print grid.materialized_array.tolist()
        expect = [
            [90080.0, 90100.0, 90120.0, 90140.0],
            [100080.0, 100100.0, 100120.0, 100140.0],
            [110080.0, 110100.0, 110120.0, 110140.0],
            [120080.0, 120100.0, 120120.0, 120140.0],
            [130080.0, 130100.0, 130120.0, 130140.0]]
        assert np.allclose(grid.materialized_array, expect)
