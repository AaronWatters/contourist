
from . import triangulated
from . import field2d
import numpy as np
import bisect


class Multiple2DContourGrid(object):

    def __init__(self, function_grid, values, segment_endpoints=()):
        self.grid = function_grid
        self.values = list(sorted(values))
        self.segment_end_points = segment_endpoints
        self.value_to_endpoints = None
        self.value_to_contour_sequences = None

    def get_contours_dictionary(self):
        """
        Return a mapping of {value: contour_sequences, ...}
        for each of the values, giving contour sequences intersecting
        the segment end_points.
        """
        self.classify_endpoints()
        self.value_to_contour_sequences = value_to_sequences = {}
        for value in self.value_to_endpoints:
            endpoints = self.value_to_endpoints[value]
            contour_maker = triangulated.DxDy2DContourGrid(self.grid, value, endpoints)
            sequences = contour_maker.get_contour_sequences()
            value_to_sequences[value] = sequences
        return value_to_sequences

    def classify_endpoints(self):
        values = self.values
        value_to_endpoints = self.value_to_endpoints = {value: [] for value in values}
        endpoints = self.segment_end_points
        for (start_point, endpoint) in endpoints:
            self.classify_endpoint(start_point, endpoint)
        # if any value has no endpoint then default to exhaustive grid search for crossings
        no_crossings = [value for value in values if len(value_to_endpoints[value]) == 0]
        if no_crossings:
            self.search_grid_for_crossings()
        return value_to_endpoints

    def classify_endpoint(self, startpoint, endpoint):
        f = self.grid.function
        f_start = f(*startpoint)
        f_end = f(*endpoint)
        return self.classify_endpoint_values(startpoint, f_start, endpoint, f_end)

    def classify_endpoint_values(self, startpoint, f_start, endpoint, f_end):
        if f_end < f_start:
            (startpoint, f_start, endpoint, f_end) = (endpoint, f_end, startpoint, f_start)
        values = self.values
        start_index = bisect.bisect_left(values, f_start)
        end_index = bisect.bisect_right(values, f_end)
        if end_index > start_index:
            value_to_endpoints = self.value_to_endpoints
            segment = (startpoint, endpoint)
            for value_index in range(start_index, end_index):
                value = values[value_index]
                value_to_endpoints[value].append(segment)

    def search_grid_for_crossings(self):
        grid = self.grid
        fij = grid.materialize_array()
        (width, height) = fij.shape
        for i in range(width - 1):
            for j in range(height - 1):
                startpoint = grid.from_grid_coordinates((i, j))
                f_start = fij[i, j]
                for (di, dj) in ((0, 1), (1,0)):
                    endgrid = (i + di, j + dj)
                    endpoint = grid.from_grid_coordinates(endgrid)
                    f_end = fij[endgrid]
                    self.classify_endpoint_values(startpoint, f_start, endpoint, f_end)


class Multiple2DContour(Multiple2DContourGrid):

    def __init__(self, xmin, ymin, xmax, ymax, dx, dy, function, values, segment_endpoints=()):
        function_grid = field2d.Function2DGrid(xmin, ymin, xmax, ymax, dx, dy, function)
        Multiple2DContourGrid.__init__(self, function_grid, values, segment_endpoints)

class Percentile2DContour(Multiple2DContourGrid):

    def __init__(self, xmin, ymin, xmax, ymax, dx, dy, function, breakpoints=10, segment_endpoints=()):
        function_grid = field2d.Function2DGrid(xmin, ymin, xmax, ymax, dx, dy, function)
        self.function_grid = function_grid
        values = self.values = self.get_values(breakpoints)
        Multiple2DContourGrid.__init__(self, function_grid, values, segment_endpoints)

    def get_values(self, breakpoints):
        samples = self.function_grid.materialize_array()
        samples = np.sort(samples.flatten())
        (nsamples,) = samples.shape
        skip = int(nsamples/breakpoints)
        values = [samples[index] for index in range(skip, nsamples, skip)]
        return values

class Linear2DContour(Percentile2DContour):

    def get_values(self, breakpoints):
        samples = self.function_grid.materialize_array()
        minimum = samples.min()
        maximum = samples.max()
        offset = (maximum - minimum) * (1.0/breakpoints)
        values = [offset * i for i in range(1, breakpoints)]
        return values
