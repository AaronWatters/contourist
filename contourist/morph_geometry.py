
import numpy as np
import surface_geometry

class MorphTriangles(object):

    def __init__(self, points4d, segment_point_indices, triangle_segment_indices):
        self.points4d = points4d = np.array(points4d)
        t_values = points4d[:, -1]
        self.max_value = t_values.max()
        self.min_value = t_values.min()
        oriented_segment_point_indices = []
        for (i,j) in segment_point_indices:
            if points4d[i][-1] > points4d[j][-1]:
                oriented_segment_point_indices.append((j,i))
            else:
                oriented_segment_point_indices.append((i,j))
        self.segment_point_indices = oriented_segment_point_indices
        self.triangle_segment_indices = triangle_segment_indices
        min_t_triangle_order = []
        self.triangle_max_t = None
        self.triangle_min_t = None

    def from_grid_coordinates(self, grid):
        points4d = grid.from_grid_coordinates(self.points4d)
        return MorphTriangles(points4d, self.segment_point_indices, self.triangle_segment_indices)

    def orient_triangles(self):
        "Orient triangles so right hand rule thumb points outward for all value of t."
        self.compute_triangle_stats()
        points4d = self.points4d
        segment_point_indices = self.segment_point_indices
        pair_3d_midpoints = [0.5 * (points4d[i1][:3] + points4d[i2][:3]) for (i1, i2) in segment_point_indices]
        triangles = self.triangle_segment_indices
        triangle_orienter = surface_geometry.SurfaceGeometry(pair_3d_midpoints, triangles)
        compatible = self.time_compatible_triangles
        oriented_triangle_indices = list(triangle_orienter.orient_triangles(compatible))
        self.triangle_segment_indices = oriented_triangle_indices

    def time_compatible_triangles(self, triangle1, triangle2):
        "two triangles are compatible if they share an extent in time."
        triangle_max_t = self.triangle_max_t
        triangle_min_t = self.triangle_min_t
        low_value = max(triangle_min_t[triangle1], triangle_min_t[triangle2])
        high_value = min(triangle_max_t[triangle1], triangle_max_t[triangle2])
        return (low_value < high_value)

    def compute_triangle_stats(self):
        "compute applicable t range for each triangle."
        points4d = self.points4d
        max_value = self.max_value
        min_value = self.min_value
        triangle_max_t = {}
        triangle_min_t = {}
        segments = self.segment_point_indices
        for triangle in self.triangle_segment_indices:
            t_min = min_value
            t_max = max_value
            for segment_index in triangle:
                (i_low, i_high) = segments[segment_index]
                low_t = points4d[i_low, -1]
                high_t = points4d[i_high, -1]
                t_min = max(t_min, low_t)   # restrict the range!
                t_max = min(t_max, high_t)
            triangle_max_t[triangle] = t_max
            triangle_min_t[triangle] = t_min
        self.triangle_max_t = triangle_max_t
        self.triangle_min_t = triangle_min_t

    def to_json(self, min_value=None, max_value=None, maxint=999999, epsilon=1e-4):
        L = []
        a = L.append
        points = self.points4d
        a("{\n")
        a('"description": "Ordered 4d morphing triangles.",\n')
        if min_value is None:
            min_value = self.min_value
        else:
            min_value = max(min_value, self.min_value)
        if max_value is None:
            max_value = self.max_value
        else:
            max_value = min(max_value, self.max_value)
        a('"max_value": %s,\n' % (max_value,))
        a('"min_value": %s,\n' % (min_value))
        points = np.array(self.points4d)
        segments = self.segment_point_indices
        triangles = self.triangle_segment_indices
        a('"counts": [%s, %s, %s],\n' % (len(points), len(segments), len(triangles),))
        maxima = points.max(axis=0)
        minima = points.min(axis=0)
        diff = np.maximum(maxima - minima, epsilon)
        a('"shift": [%s, %s, %s, %s],\n' % tuple(minima))
        scale = diff / maxint
        a('"scale": [%s, %s, %s, %s],\n' % tuple(scale))
        invscale = (1.0/scale).reshape((1,4))
        positions = ((points - minima.reshape(1,4)) * invscale).astype(np.int)
        a('"positions": %s,\n' % (flatten_json_list(positions),))
        a('"segments": %s,\n' % (flatten_json_list(segments),))
        a('"triangles": %s\n' % (flatten_json_list(triangles),))
        #a('"triangle_order": %s,\n' % (flatten_json_list(self.min_t_triangle_order),))
        #a('"triangle_max_t": %s,\n' % (flatten_json_list(self.triangle_max_t.items()),))
        a("}")
        return "".join(L)

def flatten_json_list(sequence, fmt=str):
    return "[%s]" % (",\n".join(",".join(fmt(y) for y in x) for x in sequence),)

class MorphGeometry(object):

    def __init__(self, min_value, max_value, vertices4d):
        assert min_value < max_value
        self.min_value = min_value
        self.max_value = max_value
        self.vertices4d = vertices4d
        self.mid_value = 0.5 * (min_value + max_value)
        self.triangle_4d_pairs = set()
        self.pair_4d_interpolation = {}
        self.pair_value_interpolation = {}
        nvertices = len(vertices4d)
        self.start_vertices = {}
        self.end_vertices = {}

    def triangulate_tetrahedron_at_midpoints(self, tetrahedron, epsilon=1e-4):
        vertices4d = self.vertices4d
        breakpoints = sorted(vertices4d[i][-1] for i in tetrahedron)
        previous = None
        for current in breakpoints:
            if previous is not None and (current - previous) > epsilon:
                midpoint = 0.5 * (current + previous)
                self.add_tetrahedron(tetrahedron, midpoint)
            previous = current

    def add_tetrahedron(self, tetrahedron, mid_value=None):
        (a, b, c, d) = sorted(tetrahedron)
        if mid_value is None:
            mid_value = self.mid_value
        pair_interpolations = {}
        triangle_4d_pairs = self.triangle_4d_pairs
        for pair in ((a,b), (a,c), (a,d), (b,c), (b,d), (c,d)):
            interpolation = self.interpolate_pair_3d(mid_value, *pair)
            #print "interp", pair, interpolation
            if interpolation is not None:
                p_interpolation = interpolation[0]
                pair_interpolations[pair] = p_interpolation
        #print "interpolations", pair_interpolations
        interpolated = list(pair_interpolations.keys())
        ninterpolated = len(interpolated)
        if ninterpolated == 3:
            triangle_pairs = frozenset(interpolated)
            triangle_4d_pairs.add(triangle_pairs)
        elif ninterpolated == 4:
            pair1 = interpolated[0]
            pair2 = None
            for pair in interpolated[1:]:
                if not (set(pair) & set(pair1)):
                    pair2 = pair
            assert pair2 is not None
            for pair in interpolated:
                if pair != pair1 and pair != pair2:
                    triangle_pairs = frozenset([pair1, pair2, pair])
                    triangle_4d_pairs.add(triangle_pairs)
        else:
            return None   # no triangle intersections.
        # temp
        #self.check_interpolation(pair_interpolations)
        self.pair_4d_interpolation.update(pair_interpolations)
        return pair_interpolations

    def check_interpolation(self, pair_interpolation):
        for pair in pair_interpolation:
            (i0, i1) = pair
            for i in pair:
                assert type(i) is int, repr(i)
            interp = pair_interpolation[pair]
            assert len(interp) == 3, repr(interp)

    def interpolate_pair_3d(self, value, vertex_index1, vertex_index2, epsilon=1e-5, force=False):
        pair_value_interpolation = self.pair_value_interpolation
        if vertex_index1 > vertex_index2:
            (vertex_index1, vertex_index2) = (vertex_index2, vertex_index1)
        key = (value, vertex_index1, vertex_index2)
        if key in pair_value_interpolation:
            return pair_value_interpolation[key]
        vertices4d = self.vertices4d
        vertex1 = vertices4d[vertex_index1]
        vertex2 = vertices4d[vertex_index2]
        #if np.allclose(vertex1, vertex2):
        #    return None
        v1 = vertex1[-1]
        v2 = vertex2[-1]
        if v1 > v2:
            (vertex1, v1, vertex2, v2) = (vertex2, v2, vertex1, v1)
        if value + epsilon < v1 or value - epsilon > v2:
            if force:
                # attach to vertex with nearest value.
                if abs(value - v1) < abs(value - v2):
                    value = v1
                else:
                    value = v2
            else:
                pair_value_interpolation[key] = None
                return None
        ratio = 0.0  # arbitrarily pick vertex1 on ambiguity
        diff = v2 - v1
        if diff > epsilon: #not np.allclose(v1, v2):
            ratio = (value - v1) * 1.0 / diff
        v1_3d = vertex1[:-1]
        v2_3d = vertex2[:-1]
        vertex3d = v1_3d + ratio * (v2_3d - v1_3d)
        result = (vertex3d, ratio)
        pair_value_interpolation[key] = result
        return result

    def get_start_and_end_surface_geometries(self):
        triangle_4d_pairs = self.triangle_4d_pairs
        pair_4d_interpolation = self.pair_4d_interpolation
        active_pairs = {}
        for triangle_pairs in triangle_4d_pairs:
            for pair in triangle_pairs:
                active_pairs[pair] = pair_4d_interpolation[pair]
        pair_order = active_pairs.keys()
        mid_vertices = [active_pairs[pair] for pair in pair_order]
        pair_to_index = {pair: index for (index, pair) in enumerate(pair_order)}
        mid_triangles = set()
        for triangle_pairs in triangle_4d_pairs:
            mid_triangle = frozenset(pair_to_index[pair] for pair in triangle_pairs)
            mid_triangles.add(mid_triangle)
        mid_geometry = surface_geometry.SurfaceGeometry(mid_vertices, mid_triangles)
        #mid_geometry.clean_triangles()
        mid_geometry.orient_triangles()
        clean_index_map = mid_geometry.vertex_map
        index_inverse = {clean_index_map[i]: i for i in clean_index_map}
        mid_triangles = mid_geometry.oriented_triangles
        start_geometry = self.interpolate_geometry(self.min_value, index_inverse, mid_triangles, pair_order)
        end_geometry = self.interpolate_geometry(self.max_value, index_inverse, mid_triangles, pair_order)
        return (start_geometry, end_geometry)

    def interpolate_geometry(self, value, index_inverse, triangles, pair_order):
        #assert len(index_inverse) == max(index_inverse.keys()) + 1, repr((len(index_inverse), max(index_inverse.keys())))
        vertices = [None] * len(index_inverse)
        #max_index = max(index_inverse.keys()) 
        #vertices = [np.array([0, 0, 0, 0], dtype=np.float)] * (max_index + 1)
        for index in index_inverse:
            pair_index = index_inverse[index]
            pair = pair_order[pair_index]
            interpolation = self.interpolate_pair_3d(value, pair[0], pair[1], force=True)
            if interpolation is not None:
                vertices[index] = interpolation[0]
            else:
                raise ValueError("could not interpolate " + repr(pair))
        for vertex in vertices:
            assert vertex is not None
        return surface_geometry.SurfaceGeometry(vertices, triangles)

    def json_data(self, integral=True, lists_only=True, epsilon=1e-5, maxint=9999):
        (start_geometry, end_geometry) = self.get_start_and_end_surface_geometries()
        D = {}
        D["description"] = "Morphing triangularization."
        start_positions = start_geometry.vertices
        end_positions = end_geometry.vertices
        triangles = start_geometry.triangles
        if integral:
            try:
                start_positions = np.array(start_positions, dtype=np.float)
            except ValueError:
                for x in start_positions:
                    print x
                raise
            end_positions = np.array(end_positions, dtype=np.float)
            positions = np.vstack([start_positions, end_positions])
            maxima = positions.max(axis=0)
            minima = positions.min(axis=0)
            diff = np.maximum(maxima - minima, epsilon)
            D["shift"] = list(minima)
            scale = diff / maxint
            D["scale"] = list(scale)
            invscale = (1.0/scale).reshape((1,3))
            start_positions = ((start_positions - minima) * invscale).astype(np.int)
            end_positions = ((end_positions - minima) * invscale).astype(np.int)
        D["start_positions"] = start_positions
        D["end_positions"] = end_positions
        D["triangles"] = triangles
        if lists_only:
            for slot in ("start_positions", "end_positions", "triangles"):
                D[slot] = [list(x) for x in D[slot]]
        D["min_value"] = self.min_value
        D["max_value"] = self.max_value
        return D

    def to_json(self):
        L = []
        a = L.append
        D = self.json_data(integral=True, lists_only=False)
        a("{")
        a('"description": "Morphing triangularization.",\n')
        a('"max_value": %s,\n' % (self.max_value,))
        a('"min_value": %s,\n' % (self.min_value,))
        a('"scale": %s,\n' % (D["scale"],))
        a('"shift": %s' % (D["shift"],))
        for slot in ("start_positions", "end_positions", "triangles"):
            slot_collection = D[slot]
            #string_list = [str(list(x)) for x in slot_collection]
            string_list = ["%s" % (",".join(str(y) for y in x)) for x in slot_collection]
            formatted = "[%s]" % (",".join(string_list))
            a(',\n"%s": %s' % (slot, formatted))
        a("}\n")
        return "".join(L)

def test():
    min_value = 5.1
    max_value = 14.9
    vertices = np.array([(0,0,1,0),(0,1,0,0),(1,0,0,1),(0,0,0,1),], dtype=np.float) * 10 + 5
    G = MorphGeometry(min_value, max_value, vertices)
    G.add_tetrahedron([0,1,2,3])
    print (G.to_json())

def test2():
    import pentatopes
    vertices = np.array([(0,0,0,0), (1,0,0,0), (0,1,0,1), (1,1,1,1)], dtype=np.float)
    G = MorphGeometry(0.0, 1.0, vertices)
    G.add_tetrahedron([0,1,2,3])
    C = pentatopes.GridContour4D([0]*4, None, 0.0, [])
    return C.to_json([G])

def testMt():
    import pentatopes
    def f(x,y,z,t):
        return x + t
    endpoints = [([0] * 4, [1] * 4)]
    C = pentatopes.GridContour4D([2]*4, f, 1.0, endpoints)
    C.find_tetrahedra()
    Mt = C.collect_morph_triangles()
    print (Mt.to_json())

def testMorphTriangles():
    points4d = [(0,0,0,0), (0,0,1,0), (2,3,2,3), (3,2,3,5)]
    segment_point_indices = [(0,1), (1,2), (0,2), (1,3)]
    triangle_segment_indices = [(0,1,2), (0,2,3)]
    Mt = MorphTriangles(points4d, segment_point_indices, triangle_segment_indices)
    print (Mt.to_json())

if __name__ == "__main__":
    #test()
    #print (test2())
    #testMorphTriangles()
    testMt()
