
import numpy as np

class SurfaceGeometry(object):

    def __init__(self, vertices, triangles):
        self.input_vertices = vertices
        self.input_triangles = triangles
        self.vertices = vertices
        self.triangles = triangles
        self.oriented_triangles = triangles
        self.vertex_map = tuple(range(len(vertices)))

    def clean_triangles(self):
        "Remove area 0 triangles and duplicate vertices on area 0 triangles"
        list_of_points = self.input_vertices
        set_of_triangle_indices = self.input_triangles
        keep_triangles = set()
        keep_vertices = []
        vertex_map = {}
        def new_vertex_index(vertex_index):
            if vertex_index in vertex_map:
                return vertex_map[vertex_index]
            else:
                vertex = list_of_points[vertex_index]
                result = len(keep_vertices)
                vertex_map[vertex_index] = result
                keep_vertices.append(vertex)
                return result
        for triangle in set_of_triangle_indices:
            orientation = list(triangle)
            [A, B, C] = [list_of_points[i] for i in orientation]
            cross = np.cross(A - C, B - C)
            if np.allclose(cross, 0):
                #print "  omitting triangle", orientation
                [a, b, c] = orientation
                # empty triangle: omit triangle and merge identical vertices.
                for (i, j) in [(a, b), (a, c), (b, c)]:
                    if np.allclose(list_of_points[i], list_of_points[j]):
                        # merge vertices
                        #print "      merging vertices", i, j
                        merged = new_vertex_index(i)  # == vertex_map[i]
                        vertex_map[j] = merged
            else:
                new_triangle = frozenset(new_vertex_index(i) for i in orientation)
                keep_triangles.add(new_triangle)
        self.vertices = keep_vertices
        self.triangles = keep_triangles
        self.vertex_map = vertex_map
        return (keep_vertices, keep_triangles)

    def orient_triangles(self):
        "Orient triangles so cross product of triangle vectors points outwards."
        list_of_points = self.vertices
        set_of_triangle_indices = self.triangles
        #print "orienting", len(set_of_triangle_indices), "triangles"
        segments_to_triangles = {}
        triangle_orientations = {}
        points_to_triangles_indices = {}
        for triangle in set_of_triangle_indices:
            for i in triangle:
                points_to_triangles_indices.setdefault(i, set()).add(triangle)
            (a, b, c) = triangle
            for edge in ((a,b), (b,c), (a,c)):
                s_edge = frozenset(edge)
                segments_to_triangles.setdefault(s_edge, set()).add(triangle)
        unoriented_triangles = set(set_of_triangle_indices)
        while unoriented_triangles:
            #print "unoriented", len(unoriented_triangles)
            #vertex_indices = set(p for p in triangle for triangle in unoriented_triangles)
            vertex_indices = set()
            for triangle in unoriented_triangles:
                for p in triangle:
                    vertex_indices.add(p)
            # find a triangle with max x value for some point with non-zero cross product x component
            (max_x, max_index) = max((list_of_points[i][0], i) for i in vertex_indices)
            max_x_triangles = set(triangle for triangle in points_to_triangles_indices[max_index] if triangle in unoriented_triangles)
            #if len(unoriented_triangles) < 10:
                #print "unoriented", unoriented_triangles
                #print "vertices", vertex_indices
                #print "max_x", max_x, max_index
                #print points_to_triangles_indices[max_index]
            #print "max_x_triangles", max_x_triangles
            initial_triangle = None
            maxdotx = 0
            for triangle in max_x_triangles:
                (a, b, c) = [list_of_points[i] for i in triangle]
                dotx = np.cross(a - b, a - c)[0]
                if abs(dotx) >= abs(maxdotx):
                    maxdotx = dotx
                    initial_triangle = triangle
            #print "initial triangle", triangle
            # orient the initial triangle
            orientation = tuple(initial_triangle)
            #next_triangles = set(orientation)
            (a, b, c) = [list_of_points[i] for i in orientation]
            dotx = np.cross(a - b, a - c)[0]
            # make sure dotx is *positive* (???)
            if dotx < 0:
                orientation = tuple(reversed(orientation))
            def same_orientation(orientation1, orientation2):
                if orientation1 == orientation2:
                    return True
                (a, b, c) = orientation1
                return ((b, c, a) == orientation2) or ((c, a, b) == orientation2)
            stack = [(initial_triangle, orientation)]
            def orient_edge(index1, index2, from_triangle):
                edge = frozenset((index1, index2))
                triangles = segments_to_triangles[edge]
                #assert len(triangles) <= 2, repr(list(triangles))
                if len(triangles) > 2:
                    #print "ambiguous edge?", triangles
                    pass
                for triangle in triangles:
                    if triangle != from_triangle:
                        [index3] = triangle - edge
                        orientation = (index1, index2, index3)
                        if triangle in triangle_orientations:
                            orientation1 = triangle_orientations[triangle]
                            if not same_orientation(orientation1, orientation):
                                #print "    bad orientation?", orientation1, orientation
                                pass
                        else:
                            stack.append((triangle, orientation))
            while stack:
                (triangle, orientation) = stack.pop()
                #Sprint "orientation", triangle, orientation
                triangle_orientations[triangle] = orientation
                if triangle in unoriented_triangles:
                    unoriented_triangles.remove(triangle)
                [a, b, c] = orientation
                orient_edge(c, b, triangle)
                orient_edge(b, a,  triangle)
                orient_edge(a, c, triangle)
        self.oriented_triangles = list(sorted(triangle_orientations.values()))
        return self.orient_triangles
