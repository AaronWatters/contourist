"""
Grid context for a function over n dimensions.
"""
# XXXX generalization of field2d -- should replace it

import numpy as np

class FunctionGrid(object):

    def __init__(self, mins, maxes, delta, function, materialize=False, cache=False):
        self.mins = np.array(mins, dtype=float)
        self.maxes = np.array(maxes, dtype=float)
        self.delta = np.array(delta, dtype=float)
        (self.dimension,) = self.mins.shape
        assert self.mins.shape == self.maxes.shape
        assert self.mins.shape == self.delta.shape
        self.grid_maxes = self.to_grid_vertex(self.maxes) + 1  # xxx is +1 okay?
        self.f = function
        self.cached = cache
        self.materialize = materialize
        self.materialized_array = None
        self.cache = {}
        self.grid_dimensions = (
            self.to_grid_vertex(self.maxes) + 1)
        assert np.all(self.grid_dimensions >= 2), "grid must have dimensions greater than 2"
        if materialize:
            # XXXX should sanity check dimensions too large.
            assert not cache, "do not cache and materialize at the same time."
            self.materialize_array()

    def materialize_array(self):
        shape = self.grid_dimensions
        # Can't use broadcasting so we can't use fromfunction here'
        #self.materialized_array = np.fromfunction(self.grid_function, shape, dtype=np.float)
        m = np.zeros(shape, dtype=np.float)
        f = self.grid_function
        for index_tuple in iter_indices(shape):
            m[index_tuple] = f(*index_tuple)
        self.materialized_array = m
        return m

    def to_grid_coordinates(self, xypoint):
        return (xypoint - self.mins) / self.delta

    def on_grid(self, grid_vertex):
        # xxxx boundary <= okay?
        return np.all(grid_vertex >= 0) and np.all(grid_vertex <= self.grid_maxes)

    def surrounding_vertices(self, xypoint, skip=1, grid_vertex=False):
        if grid_vertex:
            vertex0 = xypoint
        else:
            vertex0 = self.to_grid_vertex(xypoint)
        offset = np.zeros((self.dimension,), dtype=np.int)
        shifts = range(self.dimension)
        for index in range(2 ** self.dimension):
            for shift in shifts:
                offset[shift] = ((index >> shift) & 1) * skip
            yield vertex0 + offset

    def find_contour_crossing_grid_segments(self, value, skip=1):
        # should use with caching if done repeatedly.
        #print "finding crossings"
        result = []
        shape = self.grid_dimensions
        f = self.grid_function
        maxf = minf = None
        for index_tuple in iter_indices(shape, skip):
            vertex0 = np.array(index_tuple, dtype=np.int)
            f0 = f(*index_tuple)
            if maxf is None:
                maxf = minf = f0
            for vertex1 in self.surrounding_vertices(vertex0, skip, True):
                if np.sometrue(vertex0 < vertex1):
                    f1 = f(*vertex1)
                    maxf = max(f0, f1, maxf)
                    minf = min(f0, f1, minf)
                    if (f0 - value) * (f1 - value) < 0:
                        result.append((vertex0, vertex1))
        #print "found", len(result)
        return (maxf, minf, result)

    def to_grid_vertex(self, xypoint):
        return np.array(self.to_grid_coordinates(xypoint), dtype=np.int)

    def from_grid_coordinates(self, xygrid):
        xygrid = np.array(xygrid, dtype=np.float)
        delta = self.delta
        mins = self.mins
        return (xygrid * delta) + mins

    def grid_function(self, *xy_grid):
        "Contour function using grid coordinates."
        xy_grid = tuple(xy_grid)
        m = self.materialized_array
        cached = self.cached
        cache = self.cache
        #all_ints = (set([type(x) for x in xy_grid]) == set((int,)))
        all_ints = True
        for x in xy_grid:
            if not isinstance(x, int):
                all_ints = False
        if m is not None and all_ints:
            try:
                return m[xy_grid]
            except IndexError:
                pass
        if cached and all_ints and xy_grid in cache:
            return cache[xy_grid]
        xy = self.from_grid_coordinates(xy_grid)
        result = self.f(*xy)
        if cached and all_ints:
            cache[xy_grid] = result
        return result

def iter_indices(shape, skip=1):
    "generate all indices for array of shape shape."
    # xxxx This is generally useful -- should be in a library?
    lshape = len(shape)
    if lshape == 0:
        yeild ()
    elif lshape == 1:
        (n,) = shape
        for i in range(n):
            yield (i,)
    else:
        n = shape[0]
        remainder = shape[1:]
        rem_indices = list(iter_indices(remainder))
        for i in range(0, n, skip):
            head = (i,)
            for tail in rem_indices:
                yield head + tail
