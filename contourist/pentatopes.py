"""
4d contour 3d volumes built using pentatope tiling of a 4d grid.
"""

# generalized from tetrahedral.py
import tetrahedral
import morph_geometry
import surface_geometry
import itertools
import numpy as np
from numpy.linalg import norm

def generate_simplex_vertices(permutation):
    vertex = [0] * len(permutation)
    yield vertex[:]
    for index in permutation:
        vertex[index] = 1
        yield vertex[:]

def generate_pentatope_tiles():
    for permutation in itertools.permutations(range(4)):
        yield list(generate_simplex_vertices(permutation))

PENTATOPES = np.array(list(generate_pentatope_tiles()), dtype=np.int)

HYPERCUBE = np.array(
    [(i, j, k, l) for i in (0,1) for j in (0,1) for k in (0,1) for l in 0,1],
    dtype=np.int)

OFFSETS4D = np.array(
    [(i,j,k,l) 
     for i in (-1,0,1) 
     for j in (-1,0,1) 
     for k in (-1,0,1) 
     for l in (-1,0,1) 
     if i!=0 or j!=0 or k!=0 or l!=0],
    dtype=np.int)

class GridContour4D(tetrahedral.GridContour):

    box = HYPERCUBE
    offsets = OFFSETS4D

    def sanity_check(self):
        assert self.dimension == 4

    def find_tetrahedra(self):
        self.find_initial_voxels()
        while self.new_surface_voxels:
            self.expand_voxels()
        for quad in self.surface_voxels:
            self.enumerate_voxel_tetrahedra(quad)
 
    def enumerate_voxel_tetrahedra(self, quad):
        quad = np.array(quad, dtype=np.int)
        pentatopes = PENTATOPES + quad.reshape((1,1,4))
        for pentatope in pentatopes:
            #print "    tetrahedron", tetrahedron
            self.enumerate_pentatope_tetrahedra(pentatope)

    def enumerate_pentatope_tetrahedra(self, pentatope):
        low_points = set()
        high_points = set()
        values = []
        f = self.f
        value = self.value
        for p in pentatope:
            tp = tuple(p)
            pvalue = f(*tp)
            values.append(pvalue)
            # XXXX assymetry between low and high points?
            if pvalue < value:
                low_points.add(tp)
            else:
                high_points.add(tp)
        if (not low_points) or (not high_points) or np.allclose(values, value):
            # no tetrahedra
            return
        leastpoints = low_points
        mostpoints = high_points
        if len(leastpoints) > len(mostpoints):
            (leastpoints, mostpoints) = (mostpoints, leastpoints)
        #print "least, most", leastpoints, mostpoints, value
        if len(leastpoints) == 1:
            #return # DEBUG!!
            [a] = leastpoints
            [b, c, d, e] = mostpoints
            self.add_simplex((a,b), (a,c), (a,d), (a, e))
        else:
            # xxxxxx not sure about this!
            #return #  DEBUG!!!
            #raise ValueError("DISABLED FOR DEBUG")
            [a, b] = leastpoints
            [c, d, e] = mostpoints
            #[a, b] = self.sort_index_pairs(leastpoints)
            #[c, d, e] = self.sort_index_pairs(mostpoints)
            #self.add_simplex((a,d), (a,c), (b,c))
            #self.add_simplex((a,d), (b,d), (b,c))
            ac = (a, c)
            ad = (a, d)
            ae = (a, e)
            bc = (b, c)
            bd = (b, d)
            be = (b, e)
            # original
            #self.add_simplex(ac, be, ad, bc)
            #self.add_simplex(ac, be, bc, ae)
            #self.add_simplex(ac, be, ae, bd)
            #self.add_simplex(ac, be, bd, ad)
            # complete
            #self.add_simplex(ac,ad,ae,bc) # **
            #self.add_simplex(ac,ad,ae,bd) # ??
            #self.add_simplex(ac,ad,ae,be)
            #self.add_simplex(ac,ad,bc,bd) # ??
            #self.add_simplex(ac,ad,bc,be)
            #self.add_simplex(ac,ad,bd,be)
            #self.add_simplex(ac,ae,bc,bd)
            #self.add_simplex(ac,ae,bc,be) # ?? **
            #self.add_simplex(ac,ae,bd,be)
            #self.add_simplex(ac,bc,bd,be)
            #self.add_simplex(ad,ae,bc,bd)
            #self.add_simplex(ad,ae,bc,be)
            #self.add_simplex(ad,ae,bd,be) # ?? **
            #self.add_simplex(ad,bc,bd,be)
            #self.add_simplex(ae,bc,bd,be) # **
            # exp
            self.add_simplex(ac,be,ad,bd) # **
            self.add_simplex(ac,be,ad,ae) # **
            self.add_simplex(ac,be,bd,bc) # **

    def sort_index_pairs(self, pairs):
        interp = self.interpolated_contour_pairs
        order = sorted((tuple(interp[pair]), pair) for pair in pairs)
        return [x[1] for x in order]

    def order_tetrahedra(self, pair_to_index):
        order_pairs = set()
        interp = self.interpolated_contour_pairs
        global_max = None
        for pair_set in self.simplex_sets:
            values = [interp[p][-1] for p in pair_set]
            min_value = min(values)
            max_value = max(values)
            index_set = frozenset(pair_to_index[pair] for pair in pair_set)
            order_pairs.add((min_value, index_set))
            if global_max is None:
                global_max = max_value
            else:
                global_max = max(max_value, global_max)
        return (sorted(order_pairs), global_max)

    def collect_morph_triangles(self, epsilon=1e-7):
        # 4d grid edge pairs to edge interpolation: (I^4, I^4) --> R^4
        interp = self.interpolated_contour_pairs
        # ordering of grid edge pairs: [(I^4, I^4)...]
        pair_order = list(interp.keys())
        # ordering inverse map for grid edge pairs: (I^4, I^4) --> int
        pair_to_index = {frozenset(pair): index for (index, pair) in enumerate(pair_order)}
        # ordered interpolation for pairs [R^4, ...]
        vertices4d = np.array([interp[pair] for pair in pair_order])
        triangle_collector = morph_geometry.MorphGeometry(0, 1, vertices4d)
        # Set of 4-sets of pairs set(frozenset((I^4, I^4)^4))
        tetrahedra = self.simplex_sets
        # Set of index quadruples set(frozenset(I^4))
        index_tetrahedra = set(frozenset(pair_to_index[frozenset(pair)] for pair in tetrahedron) for tetrahedron in tetrahedra)
        for tetrahedron in index_tetrahedra:
            #print "triangulating", tetrahedron
            triangle_collector.triangulate_tetrahedron_at_midpoints(tetrahedron)
        #pair_3d_midpoints = [0.5 * (np.array(a[:3]) + np.array(b[:3])) for (a, b) in pair_order]
        # set of triples of pairs of indices into vertices4d (or pair_order) set(frozenset((I^2)^2))
        triangles_pairs = set(triangle_collector.triangle_4d_pairs)
        # ??? remove triangles with any segment that has 0 time extent
        #print "triangle_pairs", list(triangles_pairs)[:2]
        t_values = vertices4d[:,-1]
        t_epsilon = epsilon * (t_values.max() - t_values.min())
        #print "t_epsilon", t_epsilon
        for triangle in list(triangles_pairs):
            remove = False
            for (i1, i2) in triangle:
                diff = vertices4d[i1][-1] - vertices4d[i2][-1]
                if abs(diff) <= t_epsilon:
                    remove = True
                    break
            if remove:
                print "removing", triangle
                triangles_pairs.remove(triangle)
        # ... end of triangle removal experimental code ...
        # set of pair of indices into vertices4d set(I^2)
        segment_point_index_set = set()
        for pairs in triangles_pairs:
            segment_point_index_set.update(pairs)
        # list of pairs of indices into vertices4d: [I^2...]
        segment_point_indices = list(segment_point_index_set)
        # index lookup for pairs: 
        segment_pair_to_index = {segment_pair: index for (index, segment_pair) in enumerate(segment_point_indices)}
        #print "s_p_i", segment_pair_to_index.items()[:2]
        # triangle triples as indices into segment_point_indices: set(frozenset(I^3))
        triangle_segment_indices = set(frozenset(segment_pair_to_index[p] for p in triangle) for triangle in triangles_pairs)
        # 3d projection midpoint for segment pairs: [R^3...]
        pair_3d_midpoints = [0.5 * (vertices4d[i1][:3] + vertices4d[i2][:3]) for (i1, i2) in segment_point_indices]
        triangle_orienter = surface_geometry.SurfaceGeometry(pair_3d_midpoints, triangle_segment_indices)
        oriented_triangle_indices = list(triangle_orienter.orient_triangles())
        return morph_geometry.MorphTriangles(vertices4d, segment_point_indices, oriented_triangle_indices)    

    def iterate_morph_geometry(self):
        interp = self.interpolated_contour_pairs
        pair_order = list(interp.keys())
        pair_to_index = {pair: index for (index, pair) in enumerate(pair_order)}
        vertex_order = [interp[pair] for pair in pair_order]
        (order, global_max) = self.order_tetrahedra(pair_to_index)
        values = sorted(vertex[-1] for vertex in vertex_order)
        order_index = min_index = 0
        max_index = 1
        nvalues = len(values)
        norder = len(order)
        active_tetrahedra = set()
        while max_index <= nvalues:
            #print ("at max_index", max_index, nvalues)
            max_value = global_max
            if max_index < nvalues:
                if np.allclose(values[min_index], values[max_index]):
                    max_index += 1
                    continue
                else:
                    max_value = values[max_index]
            min_value = values[min_index]
            assert max_value >= min_value
            while order_index < norder and order[order_index][0] <= min_value:
                add_tetrahedron = order[order_index][1]
                #print "adding tetrahedron", add_tetrahedron
                active_tetrahedra.add(add_tetrahedron)
                order_index += 1
            assert len(active_tetrahedra) > 0
            if not np.allclose(min_value, max_value):
                geometry = morph_geometry.MorphGeometry(min_value, max_value, vertex_order)
                for tetrahedron in list(active_tetrahedra):
                    success = geometry.add_tetrahedron(tetrahedron)
                    if success is None:
                        # tetrahedron no longer in range.
                        active_tetrahedra.remove(tetrahedron)
                yield geometry
            # move to segment starting where the last one ended.
            min_index = max_index
            max_index += 1
        #print "done at", (max_index, nvalues)

    def json_stats(self, morphs=None):
        if morphs is None:
            morphs = list(self.iterate_morph_geometry())
        min_value = morphs[0].min_value
        max_value = morphs[-1].max_value
        return (morphs, min_value, max_value)

    def json_data(self, morphs=None):
        (morphs, min_value, max_value) = self.json_stats(morphs)
        morph_descriptions = [m.json_data() for m in morphs]
        D = {}
        D["min_value"] = min_value
        D["max_value"] = max_value
        D["morph_descriptions"] = morph_descriptions
        return D

    def to_json0(self, morphs=None):
        import json
        return json.dumps(self.json_data(morphs), indent=4)

    def to_json(self, morphs=None):
        (morphs, min_value, max_value) = self.json_stats(morphs)
        L = []
        a = L.append
        a("{\n")
        a('"description": "Sequence of morphing triangularizations.",\n')
        a('"max_value": %s,\n' % (max_value,))
        a('"min_value": %s,\n' % (min_value,))
        a('"number_of_morphs": %s,\n' % (len(morphs),))
        morph_jsons = ",\n".join(m.to_json() for m in morphs)
        a('"morph_descriptions": [\n%s]\n' % (morph_jsons,))
        a("}")
        return "".join(L)

corner = np.array([8]*4, dtype=np.int)
p1 = corner * 0.25
p2 = corner * 0.5

def test_f(*p):
    p = np.array(p, dtype=np.float)
    n0 = norm(p-p1)
    n1 = norm(p-p2)
    #return np.sin(n0+n1) + np.sin(n1 - n0/2)
    return 1.0/(.1+n0) + 1.0/(.1+n1)
    #return max(n1, n0)

def test0b():
    endpoints = [(corner, [4]*4)]
    endpoints = [(corner, p1)]
    for x in endpoints:
        for y in x:
            print y, test_f(*y)
    #stop
    value = 0.3
    G = GridContour4D(corner, test_f, value, endpoints)
    G.find_tetrahedra()
    return G

def diff(x, y, z, t):
    c = 3
    n1 = norm([x-c,z-c,(y-c)*0.25]) - t
    n2 = norm([x-c,y-c,z*0.1]) - t*t
    return min(n1, n2)

def test0d():
    corner = [6] * 4
    endpoints = [([6,6,6,0], [6,6,6,6])]
    for x in endpoints:
        for y in x:
            print y, diff(*y)
    value = -1.0
    G = GridContour4D(corner, diff, value, endpoints)
    G.find_tetrahedra()
    return G

def test0x():
    corner = [6] * 4
    endpoints = [([0]*4, [3.14]*4)]
    def f(*p):
        p = (np.array(p, dtype=np.float) - 3)
        d = np.cos(norm(p))
        q = (np.cos(p)).sum()
        return d + q
    for x in endpoints:
        for y in x:
            print y, f(*y)
    #stop
    value = 0.0
    G = GridContour4D(corner, f, value, endpoints)
    G.find_tetrahedra()
    return G

def test0c():
    corner = [10] * 4
    center = [cx, cy, cz, ct] = [1,1,1,1]
    center = [0] * 4
    def function(x,y,z,t):
        #return norm([x-cx, y-cy, t-ct])
        return norm([x-cx, y-cy, t-ct]) / (norm([0.5*(x-cx), z-cz, 0.4*(t-ct)]) + 1)
    corner = [2] * 4
    value = 5.0
    endpoints = [([0,1,0,1], center)]
    for x in endpoints:
        for y in x:
            print y, function(*y)
    G = GridContour4D(corner, function, value, endpoints)
    G.find_tetrahedra()
    return G

def test0():
    corner = [3] * 4
    def function(x,y,z,t):
        if norm([x-1, y-1, z-1, t-1]) < 0.1:
            return 1
        return 0
    def function(x,y,z,t):
        return norm([x-1,y-1,z-1,t-1])
    corner = [7] * 4
    value = 3
    endpoints = [(corner, (1,1,1,1))]
    G = GridContour4D(corner, function, value, endpoints)
    G.find_tetrahedra()
    return G

def test0p():
    corner = [2] * 4
    c = 1
    def function(x,y,z,t):
        #return x+y+z+t
        #return x + y + t
        if max(abs(x-c), abs(y-c), abs(z-c), abs(t-c)) < 0.1:
            return 2
        return 0
    value = 1
    endpoints = [([c]*4, [0]*4)]
    G = GridContour4D(corner, function, value, endpoints)
    G.find_tetrahedra()
    return G

def test():
    G = test0()
    print G.to_json()

if __name__ == "__main__":
    test()
