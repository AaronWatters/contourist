from jp_svg_canvas import cartesian_svg
from contourist import triangulated
from contourist import multiple_2d_contour
import time


class IllustrateMulti2D(object):

    def __init__(self, xmin, ymin, xmax, ymax, dx, dy, function, valueToColor, segment_endpoints=()):
        values = valueToColor.keys()
        self.driver = multiple_2d_contour.Multiple2DContour(
            xmin, ymin, xmax, ymax, dx, dy, function, values, segment_endpoints
        )
        self.dictionary = self.driver.get_contours_dictionary()
        D = self.D = cartesian_svg.doodle(xmin, ymin, xmax, ymax)
        D.show()
        for ((x0, y0), (x1, y1)) in segment_endpoints:
            D.line(None, x0, y0, x1, y1)
        for value in valueToColor:
            color = valueToColor[value]
            contours = self.dictionary[value]
            for (closed, points) in contours:
                previous = None
                if closed:
                    previous = points[-1]
                for current in points:
                    if previous is not None:
                        (x0, y0) = current
                        (x1, y1) = previous
                        D.line(None, x0, y0, x1, y1, color)
                    previous = current

COLORS = [
    "#00ffff",
    "#00eeee",
    "#00dddd",
    "#00cccc",
    "#00bbbb",
    "#00aaaa",
    "#009999",
    "#008888",
    "#007777",
    "#006666",
]

class IllustratePercentile2D(object):

    colors = COLORS

    def __init__(self, xmin, ymin, xmax, ymax, dx, dy, function):
        self.driver = self.get_driver(xmin, ymin, xmax, ymax, dx, dy, function)
        self.dictionary = self.driver.get_contours_dictionary()
        D = self.D = cartesian_svg.doodle(xmin, ymin, xmax, ymax)
        D.show()
        #for ((x0, y0), (x1, y1)) in segment_endpoints:
        #    D.line(None, x0, y0, x1, y1)
        for (value, color) in zip(self.driver.values, self.colors):
            contours = self.dictionary[value]
            for (closed, points) in contours:
                previous = None
                if closed:
                    previous = points[-1]
                for current in points:
                    if previous is not None:
                        (x0, y0) = current
                        (x1, y1) = previous
                        D.line(None, x0, y0, x1, y1, color)
                    previous = current

    def get_driver(self, xmin, ymin, xmax, ymax, dx, dy, function):
        return multiple_2d_contour.Percentile2DContour(
            xmin, ymin, xmax, ymax, dx, dy, function
        )

class IllustrateLinear2D(IllustratePercentile2D):

    def get_driver(self, xmin, ymin, xmax, ymax, dx, dy, function):
        return multiple_2d_contour.Linear2DContour(
            xmin, ymin, xmax, ymax, dx, dy, function
        )


class IllustrateDxDy2D(object):

    def __init__(self, xmin, ymin, xmax, ymax, dx, dy, function, value, segment_endpoints=None):
        self.driver = triangulated.DxDy2DContour(xmin, ymin, xmax, ymax, dx, dy, function, value, segment_endpoints)
        self.contours = self.driver.get_contour_sequences()
        D = self.D = cartesian_svg.doodle(xmin, ymin, xmax, ymax)
        D.show()
        radius = (xmax - xmin) * 0.1
        for (closed, points) in self.contours:
            previous = None
            if closed:
                previous = points[-1]
            for current in points:
                if previous is not None:
                    (x0, y0) = current
                    (x1, y1) = previous
                    D.line(None, x0, y0, x1, y1)
                previous = current

class Illustrate2d(object):

    def __init__(self, horizontal_n, vertical_m, function, value, segment_endpoints=None):
        n = self.n = horizontal_n
        m = self.m = vertical_m
        D = self.D = cartesian_svg.doodle(0, 0, m, n)
        D.rect(None, 0, 0, n, m, "cornsilk")
        D.show()
        G = self.G = triangulated.Grid2DContour(n, m, function, value, segment_endpoints, self.callback)
        self.contours = G.get_contour_sequences()

    def callback(self, G=None, contours_only=False, sleep=1.0):
        G = self.G
        D = self.D
        D.empty()
        m = self.m
        n = self.n
        radius = min(m, n) * 0.01
        if not contours_only:
            interpolated_pairs = G.interpolated_contour_pairs
            for [(x0, y0), (x1, y1)] in G.end_points:
                D.circle(None, x0, y0, radius, "cyan")
                D.circle(None, x1, y1, radius, "magenta")
                D.line(None, x0, y0, x1, y1, "#999999")
            for pair in interpolated_pairs:
                [(x0, y0), (x2, y2)] = pair
                (x1, y1) = interpolated_pairs[pair]
                D.circle(None, x0, y0, radius, "blue")
                D.circle(None, x2, y2, radius, "red")
                D.line(None, x0, y0, x1, y1, "blue")
                D.line(None, x1, y1, x2, y2, "red")
                D.circle(None, x1, y1, radius, "black")
        for (closed, sequence) in G.contours:
            previous = None
            if closed:
                previous = sequence[-1]
            for current in sequence:
                if previous is not None:
                    (x0, y0) = current
                    (x1, y1) = previous
                    D.line(None, x0, y0, x1, y1, "green", 2)
                previous = current
            for (p, c, r) in [(sequence[0], "green", 2*radius), (sequence[-1], "yellow", 1.5*radius)]:
                (x, y) = p
                D.circle(None, x, y, r, c)
        time.sleep(sleep)