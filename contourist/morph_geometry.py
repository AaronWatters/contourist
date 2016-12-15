
import numpy as np
import surface_geometry

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

    def add_tetrahedron(self, tetrahedron):
        (a, b, c, d) = sorted(tetrahedron)
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

    def add_tetrahedron0(self, tetrahedron):
        (a, b, c, d) = sorted(tetrahedron)
        mid_value = self.mid_value
        pair_interpolations = {}
        pair0 = pair1 = None
        interp0 = interp1 = None
        triangle_4d_pairs = self.triangle_4d_pairs
        for pair in ((a,b), (a,c), (a,d), (b,c), (b,d), (c,d)):
            interpolation = self.interpolate_pair_3d(mid_value, *pair)
            #print "interp", pair, interpolation
            if interpolation is not None:
                p_interpolation = interpolation[0]
                pair_interpolations[pair] = p_interpolation
                #print "comparing", (interp0, p_interpolation)
                if pair0 is None:
                    pair0 = pair
                    interp0 = p_interpolation
                elif pair1 is None and not np.allclose(interp0, p_interpolation):
                    pair1 = pair
                    interp1 = p_interpolation
        if pair0 is None:
            return None   # tetrahedron is out of range
        assert len(pair_interpolations) > 2, "bad tetrahedron: " + repr(tetrahedron)
        for pair in pair_interpolations:
            if pair != pair0 and pair != pair1:
                interpolation = pair_interpolations[pair]
                if True: #not (np.allclose(interpolation, interp0) or np.allclose(interpolation, interp1)):
                    triangle_pairs = frozenset([pair0, pair1, pair])
                    triangle_4d_pairs.add(triangle_pairs)
        #self.check_interpolation(pair_interpolation)
        self.pair_4d_interpolation.update(pair_interpolations)
        return pair_interpolations

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

if __name__ == "__main__":
    #test()
    print (test2())
