"""
Grid context for a function over 2 dimensions.
"""

import numpy as np
import grid_field

def Function2DGrid(xmin, ymin, xmax, ymax, dx, dy, function, materialize=False, cache=False):
    return grid_field.FunctionGrid((xmin, ymin), (xmax, ymax), (dx, dy), function, materialize, cache)

class Function2DGrid_xxx(object):

    def __init__(self, xmin, ymin, xmax, ymax, dx, dy, function, materialize=False, cache=False):
        self.mins = np.array((xmin, ymin), dtype=float)
        self.maxes = np.array((xmax, ymax), dtype=float)
        self.delta = np.array((dx, dy), dtype=float)
        self.f = function
        self.cached = cache
        self.materialize = materialize
        self.materialized_array = None
        self.cache = {}
        (self.horizontal_n, self.vertical_m) = self.grid_dimensions = (
            self.to_grid_vertex(self.maxes) + 1)
        assert np.all(self.grid_dimensions >= 2), "grid must have dimensions greater than 2"
        if materialize:
            assert not cache, "do not cache and materialize at the same time."
            self.materialize_array()

    def materialize_array(self):
        shape = (self.horizontal_n, self.vertical_m)
        #self.materialized_array = np.fromfunction(self.grid_function, shape, dtype=np.float)
        m = np.zeros(shape)
        for i in range(self.horizontal_n):
            for j in range(self.vertical_m):
                m[i, j] = self.grid_function(i, j)
        self.materialized_array = m
        return m

    def to_grid_coordinates(self, xypoint):
        return (xypoint - self.mins) / self.delta

    def surrounding_vertices(self, xypoint):
        vertex0 = self.to_grid_vertex(xypoint)
        yield vertex0
        for offset in ((0,1), (1,0), (1,1)):
            yield vertex0 + np.array(offset, dtype=np.int)

    def to_grid_vertex(self, xypoint):
        return np.array(self.to_grid_coordinates(xypoint), dtype=np.int)

    def from_grid_coordinates(self, xygrid):
        xygrid = np.array(xygrid, dtype=np.float)
        return (xygrid * self.delta) + self.mins

    def grid_function(self, x_grid, y_grid):
        "Contour function using grid coordinates."
        xy_grid = (x_grid, y_grid)
        m = self.materialized_array
        cached = self.cached
        cache = self.cache
        all_ints = (type(x_grid) is int and type(y_grid) is int)
        if m is not None and all_ints:
            try:
                return m[xy_grid]
            except IndexError:
                pass
        if cached and all_ints and xy_grid in cache:
            return cache[xy_grid]
        (x, y) = self.from_grid_coordinates(xy_grid)
        result = self.f(x, y)
        if cached and all_ints:
            cache[xy_grid] = result
        return result
