from jp_gene_viz import doodle3d
from jp_svg_canvas import cartesian_svg
import tetrahedral
import pentatopes
import triangulated
import numpy as np

class CDoodle(doodle3d.Doodle3D):

    def implicit_surface(self, mins, maxes, delta, f, v, color, 
            kind="solid", opacity=None, interpolate=False):
        I = tetrahedral.TriangulatedIsosurfaces(mins, maxes, delta, f, v, [], linear_interpolate=interpolate)
        I.search_for_endpoints()
        (points, triangles) = I.get_points_and_triangles()
        return self.triangle_surface(points, triangles, color, kind, opacity)

    def implicit_morph(self, mins, maxes, delta, f, v, duration, color, interpolate=True):
        C = pentatopes.MorphingIsoSurfaces(mins, maxes, delta, f, v, [], linear_interpolate=interpolate)
        C.search_for_endpoints()
        M = C.collect_morph_triangles()
        points = M.points4d
        segments = M.segment_point_indices
        triangles = M.triangle_segment_indices
        return self.morph_triangles(points, segments, triangles, mins[-1], maxes[-1], duration, color)

def add_implicit_curve(doodle2d, mins, maxes, delta, f, v, color):
    (xmin, ymin) = mins
    (xmax, ymax) = maxes
    (dx, dy) = delta
    contour_maker = triangulated.DxDy2DContour(xmin, ymin, xmax, ymax, dx, dy, f, v)
    contours = contour_maker.get_contour_sequences()
    for (closed, points) in contours:
        if closed:
            points = list(points) + [points[0]]
        points = np.array(points)
        xs = points[:, 0]
        ys = points[:, 1]
        doodle2d.sequence(None, xs, ys, color)