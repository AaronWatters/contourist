"""
Triangulated 2d contours.
"""

# xxxx add caching for expensive function

import numpy as np
import field2d

adjacent_offsets = [
    (0,1), (1,1), (1,0), (0,-1), (-1,-1), (-1,0),
]

adjacency_array = np.array(adjacent_offsets, dtype=np.int)

def adjacent_pairs(low_pair, high_pair):
    low_pair = np.array(low_pair, dtype=np.int)
    high_pair = np.array(high_pair, dtype=np.int)
    n_adjacencies = len(adjacent_offsets)
    low_offset = tuple(low_pair - high_pair)
    low_index = adjacent_offsets.index(low_offset)
    high_offset = tuple(high_pair - low_pair)
    high_index = adjacent_offsets.index(high_offset)
    for (hshift, lshift) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        adjacent_low = high_pair + adjacency_array[(low_index + lshift) % n_adjacencies]
        adjacent_high = low_pair + adjacency_array[(high_index + hshift) % n_adjacencies]
        yield (tuple(adjacent_low), tuple(adjacent_high))

class ContourGrid(object):

    "Shared functionality for 2d and 3d"

    def __init__(self, function_grid, value, segment_endpoints=None, linear_interpolate=True):
        """
        Derive piecewise approximate contours for function(x,y) = value
        starting at points (xmin + i*dx, ymin + j*dy) in range (xmin...xmax, ymin...ymax)
        """
        self.linear_interpolate = linear_interpolate
        self.grid = function_grid
        self.value = value
        self.segment_endpoints = segment_endpoints
        grid_endpoints = None
        if segment_endpoints is not None:
            grid_endpoints = []
            for (start_xy, end_xy) in segment_endpoints:
                grid_endpoint = self.to_grid_endpoint(start_xy, end_xy)
                if grid_endpoint is not None:
                    grid_endpoints.append(grid_endpoint)
            if len(grid_endpoints) < 1:
                # default to grid search
                grid_endpoints = None
        self.contour_maker = self.get_contour_maker(grid_endpoints)
        #self.contour_maker = Grid2DContour(function_grid.horizontal_n, function_grid.vertical_m, 
        #    function_grid.grid_function, self.value, grid_endpoints)
        self.grid_values = None

    def to_grid_endpoint(self, start_xy, end_xy):
        grid = self.grid
        value = self.value
        for start_grid in grid.surrounding_vertices(start_xy):
            for end_grid in grid.surrounding_vertices(end_xy):
                if not np.all(start_grid == end_grid):
                    if (grid.grid_function(*start_grid)-value) * (grid.grid_function(*end_grid)-value) <= 0:
                        return (start_grid, end_grid)
        # default
        return None


class DxDy2DContourGrid(ContourGrid):

    def get_contour_maker(self, grid_endpoints):
        assert self.linear_interpolate, "non-linear interpolation not implemented yet for 2d"
        grid = self.grid
        (horizontal_n, vertical_m) = grid.grid_dimensions
        f = grid.grid_function
        return Grid2DContour(horizontal_n, vertical_m, f, self.value, grid_endpoints)

    def get_contour_sequences(self):
        self.grid_contours = self.contour_maker.get_contour_sequences()
        self.contours = [self.from_grid_contour(c) for c in self.grid_contours]
        return self.contours

    def from_grid_contour(self, contour):
        (closed, grid_points) = contour
        grid = self.grid
        xy_points = [grid.from_grid_coordinates(grid_point) for grid_point in grid_points]
        return (closed, xy_points)


class DxDy2DContour(DxDy2DContourGrid):

    def __init__(self, xmin, ymin, xmax, ymax, dx, dy, function, value, segment_endpoints=None):
        function_grid = field2d.Function2DGrid(xmin, ymin, xmax, ymax, dx, dy, function)
        DxDy2DContourGrid.__init__(self, function_grid, value, segment_endpoints)

class Grid2DContour(object):

    def __init__(self, horizontal_n, vertical_m, function, value, segment_endpoints=None, callback=None):
        """
        Derive piecewise approximate countours for function(x,y) == value
        starting at points found between segment endpoints inside the grid 
        (0 .. horizontal_n-1, 0 .. vertical_m-1).

        Parameters
        ----------
        horizontal_n: int
            Vary integral x values from 0 .. horizontal_n-1.
        vertical_m: int
            Vary integral y value from 0 .. vertical_m-1.
        function: callable (int, int) to float
            Function to contour.
        value: float
            Value for contour
        segment_endpoints: [[(int, int), (int, int)] ...]
            [(x0, y0), (x1, y1)] where (f(x0, y0)-value) * (f(x1, y1)-value) <= 0.
            Integral end points for segments crossing the contour.
            The algorithm will produce at least one contour intersecting each
            endpoint pair provided.  Optional -- if omitted the
            algorithm will exhaustively search the adjacent grid for crossing
            grid point pairs.
        callback: callable
            Optional callback for algorithm illustration or debugging
            called as callback(self).
        """
        #self.epsilon = epsilon
        n = self.n = horizontal_n
        m = self.m = vertical_m
        self.corner = np.array([n, m], dtype=np.int)
        self.f = function
        self.z = value
        if segment_endpoints is None:
            self.search_grid()
        else:
            self.end_points = np.array(segment_endpoints, dtype=np.int)
        # cache of (x,y) --> f(x,y)
        #self.location_values = {}
        # ((x1,y1), (x2,y2)) --> (x0,y0), interpolated contour location.
        self.interpolated_contour_pairs = {}
        # set of ((x1,y1), (x2,y2)) for low-point/high-point pairs on horizon.
        self.new_contour_pairs = set()
        self.callback = callback
        # [(closed, point_sequence), ...] for contours found
        self.contours = []
        self.triangle_triples = set()

    def search_grid(self, skip=1):
        "Search the grid for additional verticals or horizontals that cross the contours."
        end_points = set()
        n = self.n
        m = self.m
        for i in range(0, n - 1, skip):
            for j in range(0, m - 1, skip):
                p0 = (i, j)
                p0a = np.array((i, j), dtype=np.int)
                for p1 in [(i + 1, j), (i, j + 1)]:
                    p1a = np.array(p1, dtype=np.int)
                    if self.contour_pair_interpolation(p0a, p1a) is not None:
                        end_points.add((p0, p1))
                    elif self.contour_pair_interpolation(p1a, p0a) is not None:
                        end_points.add((p1, p0))
        self.end_points = np.array(list(end_points), dtype=np.int)

    def check_callback(self):
        "for testing and debugging."
        callback = self.callback
        if callback:
            callback(self)

    def get_contour_sequences(self):
        """
        Return [(closed, sequence)...]
            Where closed is true if contour is closed and sequence
            is a sequence of interpolated pairs for each contour point.  Include at 
            least one contour intersecting each input segment.
        """
        contours = self.contours = []
        interpolated = self.interpolated_contour_pairs
        self.check_callback()
        self.find_initial_contour_pairs()
        self.check_callback()
        while self.new_contour_pairs:
            self.expand_contour_pairs()
            self.check_callback()
        adjacencies = self.find_adjacencies()
        edge_pairs = set(pair for pair in adjacencies if len(adjacencies[pair]) < 2)
        unvisited = set(interpolated)
        highs = {}
        lows = {}
        for pair in unvisited:
            (l, h) = pair
            highs.setdefault(h, set()).add(pair)
            lows.setdefault(l, set()).add(pair)
        def remove_pair(pair):
            if pair in unvisited:
                unvisited.remove(pair)
            if pair in edge_pairs:
                edge_pairs.remove(pair)
        while unvisited:
            new_contour = []
            if edge_pairs:
                pair = edge_pairs.pop()
                unvisited.remove(pair)
                closed = False
            else:
                pair = unvisited.pop()
                closed = True
            last_pair = None
            while pair is not None:
                # fix the orientation of each end point
                (low, high) = pair
                for high_pair in highs.get(low, []):
                    remove_pair(high_pair)
                for low_pair in lows.get(high, []):
                    remove_pair(low_pair)
                pair_interpolation = interpolated[pair]
                # append non-repeating interpolations
                if len(new_contour) == 0 or not np.allclose(new_contour[-1], pair_interpolation):
                    new_contour.append(pair_interpolation)
                next_pair = None
                for adjacent in adjacencies[pair]:
                    if adjacent in unvisited:
                        adjacent_interpolation = interpolated[adjacent]
                        remove_pair(adjacent)
                        #unvisited.remove(adjacent)
                        #if adjacent in edge_pairs:
                        #    edge_pairs.remove(adjacent)
                        next_pair = adjacent
                        break
                if last_pair is not None:
                    triple = frozenset(pair + last_pair)
                    assert len(triple) == 3
                    #print "pair", pair, "last_pair", last_pair, "triple", triple
                    self.triangle_triples.add(triple)
                last_pair = pair
                pair = next_pair
            #print "done"
            if np.allclose(new_contour[0], new_contour[-1]):
                closed = True
            contours.append((closed, np.array(new_contour)))
            self.check_callback()
        return contours

    def find_adjacencies(self):
        adjacencies = {}
        interpolated = self.interpolated_contour_pairs
        for pair in interpolated:
            adjacent = set()
            for possible in adjacent_pairs(*pair):
                if possible in interpolated:
                    adjacent.add(possible)
            assert len(adjacent) > 0, "no adjacencies " + repr(pair)
            adjacencies[pair] = adjacent
        return adjacencies

    def find_initial_contour_pairs(self):
        new_pairs = set()
        for (low_point, high_point) in self.end_points:
            if self.contour_pair_interpolation(low_point, high_point) is None:
                (low_point, high_point) = (high_point, low_point)
                assert self.contour_pair_interpolation(low_point, high_point) is not None, (
                    "bad end points " + repr((low_point, high_point))
                )
            while np.sometrue(np.abs(low_point - high_point) > 1):
                mid_point = (low_point + high_point) // 2
                if self.contour_pair_interpolation(low_point, mid_point) is not None:
                    high_point = mid_point
                else:
                    assert self.contour_pair_interpolation(mid_point, high_point) is not None
                    low_point = mid_point
            pair_to_interpolation = self.find_all_adjacent_contour_pairs(low_point, True)
            pair_to_interpolation.update(self.find_all_adjacent_contour_pairs(high_point, False))
            assert len(pair_to_interpolation) > 0
            self.interpolated_contour_pairs.update(pair_to_interpolation)
            new_pairs.update(set(pair_to_interpolation))
        self.new_contour_pairs = new_pairs

    def expand_contour_pairs(self):
        horizon_pairs = self.new_contour_pairs
        new_pairs = set()
        pair_to_interpolation = {}
        for (low_point, high_point) in horizon_pairs:
            pair_to_interpolation.update(self.find_all_adjacent_contour_pairs(low_point, True))
            pair_to_interpolation.update(self.find_all_adjacent_contour_pairs(high_point, False))
        new_pairs = set(pair for pair in pair_to_interpolation if pair not in self.interpolated_contour_pairs)
        self.interpolated_contour_pairs.update(pair_to_interpolation)
        self.new_contour_pairs = new_pairs

    def in_range(self, pair):
        return np.all(pair >= 0) and np.all(pair < self.corner)

    #def triangular_adjacent(self, pair1, pair2):
    #    diff = pair1 - pair2
    #    return tuple(diff) in adjacent_offsets

    def contour_pair_interpolation(self, low_point, high_point):
        #epsilon = self.epsilon
        f = self.f
        z = self.z
        flow = f(*low_point)
        fhigh = f(*high_point)
        interpolated = None
        if flow <= z and fhigh >= z:
            ratio = 0.5
            denominator = 1.0 * (fhigh - flow)
            if not np.allclose(denominator, 0):
                ratio = (z - flow)/denominator
            # Prevent exact landing on grid points
            #ratio = min(1.0 - epsilon, max(epsilon, ratio))
            interpolated = low_point + ratio * (high_point - low_point)
        return interpolated

    def find_all_adjacent_contour_pairs(self, location, is_low=True):
        location = np.array(location, dtype=np.int)
        offset_locations = adjacent_offsets + location.reshape((1,2))
        result = {}
        for offset_location in offset_locations:
            if self.in_range(offset_location):
                if is_low:
                    interpolated = self.contour_pair_interpolation(location, offset_location)
                    if interpolated is not None:
                        result[(tuple(location), tuple(offset_location))] = interpolated
                else:
                    interpolated = self.contour_pair_interpolation(offset_location, location)
                    if interpolated is not None:
                        result[(tuple(offset_location), tuple(location))] = interpolated
        return result

