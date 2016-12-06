"""
3d contour surfaces built using tetrahedral tiling of a 3d grid.
"""

# Based on the algorithmic sketch described here:
# http://paulbourke.net/geometry/polygonise/#tetra

# todo:
# grid search for 3d
# average inverse distance to point set function
# 

import numpy as np
import grid_field
import triangulated
import surface_geometry

# cube coordinates
A = (0, 0, 0)
B = (0, 0, 1)
C = (0, 1, 0)
D = (0, 1, 1)
E = (1, 0, 0)
F = (1, 0, 1)
G = (1, 1, 0)
H = (1, 1, 1)

CUBE = np.array([A, B, C, D, E, F, G, H], dtype=np.int)

# xxxx really should use indices?
TETRAHEDRA = np.array([
    [A, H, B, D],
    [A, H, D, C],
    [A, H, C, G],
    [A, H, G, E],
    [A, H, E, F],
    [A, H, F, B],
], dtype=np.int)

OFFSETS = np.array([(i,j,k) for i in (-1,0,1) for j in (-1,0,1) for k in (-1,0,1) if i!=0 or j!=0 or k!=0])


class Delta3DContour(triangulated.ContourGrid):

    def get_contour_maker(self, grid_endpoints):
        grid = self.grid
        (horizontal_n, vertical_m, forward_l) = grid.grid_dimensions
        f = grid.grid_function
        value = self.value
        return Grid3DContour(horizontal_n, vertical_m, forward_l, f, value, grid_endpoints)

    def search_for_endpoints(self, skip=1):
        # turn on grid caching
        grid = self.grid
        if not grid.materialize:
            grid.cached = True
        (maxf, minf, grid_endpoints) = grid.find_contour_crossing_grid_segments(self.value, skip)
        self.contour_maker = self.get_contour_maker(grid_endpoints)

    def get_points_and_triangles(self):
        (grid_points, triangles) = self.contour_maker.get_points_and_triangles()
        convert_point = self.grid.from_grid_coordinates
        points = [convert_point(p) for p in grid_points]
        return (points, triangles)

class TriangulatedIsosurfaces(Delta3DContour):

    def __init__(self, mins, maxes, delta, function, value, segment_endpoints):
        grid = grid_field.FunctionGrid(mins, maxes, delta, function)
        return Delta3DContour.__init__(self, grid, value, segment_endpoints)

class Grid3DContour(object):

    def __init__(self, horizontal_n, vertical_m, forward_l, function, value, segment_endpoints, 
        linear_interpolate=True, callback=None):
        """
        Derive surface triangularizations approximating the surface function(x,y,z) == value
        starting at points found between segment endpoints inside the grid 
        (0..horizontal_n, 0..vertical_m, 0..forward_l).

        Parameters
        ----------
        horizontal_n, vertical_m, forward_l: int
            Grid x, y, z integral extent from origin.
        function: callable (int, int, int) to float
            Function to triangulate.
        value: float
            Value at which to triangulate the function.
        segment_endpoints: [((int, int, int), (int, int, int)), ...]
            Pairs ((x0, y0, z0), (x1, y1, z1) having
            (function(x0, y0, z0) - value) * (function(x1, y1, z1) - value) <= 0.
            Integral end points for segments crossing the surface.
            The algorithm will produce at least one surface fragment intersecting each
            endpoint pair provided which crosses the contour.
        linear_interpolate: boolean
            If true then linearly interpolate the values of f at grid points.
            Otherwise evaluate f at non-grid points.
        callback: callable
            Optional callback for algorithm illustration or debugging
            called as callback(self).
        """
        self.linear_interpolate = linear_interpolate
        n = self.n = horizontal_n
        m = self.m = vertical_m
        l = self.l = forward_l
        self.end_points = segment_endpoints
        self.corner = np.array([n, m, l], dtype=np.int)
        self.f = function
        self.value = value
        # (i,j,k) vertices that cross the surface
        self.surface_voxels = set()
        self.new_surface_voxels = set()
        # voxels which have been examined
        self.visited_voxels = set()
        # Interpolation for adjacent vertices crossing the surface 
        # ((i0,j0,k0), (i1,j1,k1)) --> (x,y,z)
        # Assuming f(i0, j0, k0) <= value and f(i1,j1,k1) >= value
        self.interpolated_contour_pairs = {}
        # Set of triangles containing sets of 3 pairs of adjacent
        # vertices with interpolated point on the triangle.
        self.triangle_pair_triples = set()
        self.callback = callback

    def add_triangle(self, pair1, pair2, pair3):
        pairs = frozenset(self.interpolate_pair(pair) for pair in (pair1, pair2, pair3))
        if pairs in self.triangle_pair_triples:
            raise ValueError("duplicate triangle " + repr(pairs))
        self.triangle_pair_triples.add(pairs)

    def interpolate_pair(self, pair):
        (p0, p1) = pair
        (oriented, interpolated) = self.contour_pair_interpolation(p0, p1, True)
        self.interpolated_contour_pairs[oriented] = interpolated
        return oriented

    def check_callback(self):
        "for testing and debugging."
        callback = self.callback
        if callback:
            callback(self)

    def get_points_and_triangles(self):
        """
        Return (list_of_points, set_of_triangles_indices) for triangles approximating the surface 
             f(x,y,z) == value.
        Where list_of_points is a sequence of 3d points on the surface and set_of triangles
        is a set of set([i,j,k]) listing indices of points for each triangle on the triangulation.
        """
        self.find_initial_voxels()
        while self.new_surface_voxels:
            self.expand_voxels()
        for triple in self.surface_voxels:
            #print "enumerating", triple
            self.enumerate_voxel_triangles(triple)
        return self.extract_points_and_triangles()

    def enumerate_voxel_triangles(self, triple):
        triple = np.array(triple, dtype=np.int)
        tetrahedra = TETRAHEDRA + triple.reshape((1,1,3))
        for tetrahedron in tetrahedra:
            #print "    tetrahedron", tetrahedron
            self.enumerate_tetrahedron_triangles(tetrahedron)

    def enumerate_tetrahedron_triangles(self, tetrahedron):
        low_points = set()
        high_points = set()
        values = []
        f = self.f
        value = self.value
        for p in tetrahedron:
            tp = tuple(p)
            pvalue = f(*tp)
            values.append(pvalue)
            # XXXX assymetry between low and high points?
            if pvalue < value:
                low_points.add(tp)
            else:
                high_points.add(tp)
        if (not low_points) or (not high_points) or np.allclose(values, value):
            # no triangles
            return
        leastpoints = low_points
        mostpoints = high_points
        if len(leastpoints) > len(mostpoints):
            (leastpoints, mostpoints) = (mostpoints, leastpoints)
        if len(leastpoints) == 1:
            [a] = leastpoints
            [b, c, d] = mostpoints
            self.add_triangle((a,b), (a,c), (a,d))
        else:
            [a, b] = leastpoints
            [c, d] = mostpoints
            self.add_triangle((a,d), (a,c), (b,c))
            self.add_triangle((a,d), (b,d), (b,c))

    def extract_points_and_triangles(self):
        """
        Translate internal data structure representations to (list_of_points, set_of_triangle_indices).
        """
        geometry = self.extract_surface_geometry()
        return (geometry.vertices, geometry.oriented_triangles)

    def extract_surface_geometry(self):
        triangle_pairs_set = set()
        for triple in self.triangle_pair_triples:
            triangle_pairs_set.update(triple)
        pair_list = list(triangle_pairs_set)
        pair_number = {pair: count for (count, pair) in enumerate(pair_list)}
        list_of_points = [self.interpolated_contour_pairs[pair] for pair in pair_list]
        set_of_triangles_indices = set()
        for triangle in self.triangle_pair_triples:
            index_set = frozenset(pair_number[pair] for pair in triangle)
            set_of_triangles_indices.add(index_set)
        geometry = surface_geometry.SurfaceGeometry(list_of_points, set_of_triangles_indices)
        geometry.clean_triangles()
        geometry.orient_triangles()
        #(list_of_points, set_of_triangles_indices) = clean_triangles(list_of_points, set_of_triangles_indices)
        #set_of_triangles_indices = orient_triangles(list_of_points, set_of_triangles_indices)
        return geometry

    def border_voxel(self, vertex):
        vertex = np.array(vertex, dtype=np.int)
        voxel_grid_points = vertex.reshape((1,3)) + CUBE
        f = self.f
        function_values = [f(*p) for p in voxel_grid_points]
        # if the function values don't vary enough it's not on the border.
        value = self.value
        if np.allclose(value, function_values):
            return False
        else:
            return (min(function_values) <= value) and (max(function_values) >= value)

    def find_initial_voxels(self):
        """
        Attempt to use the segment_endpoints to find lower left corners for voxels
        containing triangles lying on the surface f(x,y,z) == value
        """
        new_voxels = set()
        f = self.f
        value = self.value
        end_points = np.array(self.end_points, dtype=np.int)
        visited = self.visited_voxels
        for (low_point, high_point) in end_points:
            low_value = f(*low_point)
            high_value = f(*high_point)
            if low_value > value or high_value < value:
                (low_point, low_value, high_point, high_value) = (high_point, high_value, low_point, low_value)
            assert (low_value <= value and high_value >= value), (
                "Bad end points " + repr((low_point, low_value, high_point, high_value, value))
            )
            while np.sometrue(np.abs(low_point - high_point) > 1):
                mid_point = (low_point + high_point) // 2
                mid_value = f(*mid_point)
                if mid_value < value:
                    low_point = mid_point
                else:
                    high_point = mid_point
            for point in (low_point, high_point):
                tpoint = tuple(point)
                if tpoint in visited:
                    continue
                visited.add(tpoint)
                if self.border_voxel(point):
                    new_voxels.add(tpoint)
                    continue
                for offset_point in (OFFSETS + point.reshape((1,3))):
                    toffset = tuple(offset_point)
                    if toffset in visited:
                        continue
                    visited.add(toffset)
                    if self.border_voxel(offset_point):
                        new_voxels.add(toffset)
                        break
        #print "initial voxels", new_voxels
        self.new_surface_voxels = new_voxels

    def expand_voxels(self):
        """
        Add more unvisited voxels containing triangles on the surface.
        """
        horizon_voxels = self.new_surface_voxels
        new_voxels = set()
        visited = self.visited_voxels
        surface = self.surface_voxels
        for voxel in horizon_voxels:
            visited.add(voxel)
            surface.add(voxel)
            avoxel = np.array(voxel, dtype=np.int)
            for aoffset in (OFFSETS + avoxel.reshape((1,3))):
                toffset = tuple(aoffset)
                if toffset not in visited and self.in_range(aoffset):
                    visited.add(toffset)
                    if self.border_voxel(aoffset):
                        new_voxels.add(toffset)
        self.new_surface_voxels = new_voxels

    def in_range(self, point):
        """
        Test whether a point is in the grid.
        """
        return np.all(point >= 0) and np.all(point < self.corner)

    def contour_pair_interpolation(self, low_point, high_point, swap=False, iterations=5):
        low_a = np.array(low_point, dtype=np.float)
        high_a = np.array(high_point, dtype=np.float)
        f = self.f
        z = self.value
        flow = f(*low_point)
        fhigh = f(*high_point)
        if swap and flow > fhigh:
            (low_point, flow, low_a, high_point, fhigh, high_a) = (
                high_point, fhigh, high_a, low_point, flow, low_a)
        interpolated = None
        if flow <= z and fhigh >= z:
            ratio = 0.5
            denominator = 1.0 * (fhigh - flow)
            if not np.allclose(denominator, 0):
                ratio = (z - flow)/denominator
            interpolated = low_a + ratio * (high_a - low_a)
        if interpolated is not None and not self.linear_interpolate:
            # iterate for better approximation to f(interpolated) == z
            count = 0
            fint = f(*interpolated)
            low = low_a
            high = high_a
            while count < iterations and not np.allclose(fint, z) and not np.allclose(low_a, high_a):
                count += 1
                if fint < z:
                    low_a = interpolated
                    flow = fint
                else:
                    assert fint > z
                    high_a = interpolated
                    fhigh = fint
                ratio = (z - flow) * 1.0 / (fhigh - flow)
                interpolated = low_a + ratio * (high_a - low_a)
                fint = f(*interpolated)
        if swap:
            assert interpolated is not None
            return ((low_point, high_point), interpolated)
        return interpolated
