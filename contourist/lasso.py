
import numpy as np

def inside_lasso(test_points, closed_lasso_path, epsilon=1e-3):
    """
    return {index: point} for points inside 
    the path points.
    """
    ordered_points = sorted((x,y,i) for (i, (x,y)) in enumerate(test_points))
    #print "ordered points", ordered_points
    index_to_point = {}
    ordered_segments = []
    previous_vertex = closed_lasso_path[-1]
    for vertex in closed_lasso_path:
        (vx, vy) = vertex
        (px, py) = previous_vertex
        if vx < px:
            segment = (vx, vy, px, py)
        else:
            segment = (px, py, vx, vy)
        ordered_segments.append(segment)
        previous_vertex = vertex
    ordered_segments = sorted(ordered_segments)
    #print "ordered segments", ordered_segments
    segment_index = 0
    active_segments = set()
    for (x,y,i) in ordered_points:
        #print "testing", (x,y,i)
        while segment_index < len(ordered_segments) and ordered_segments[segment_index][0] <= x:
            active_segments.add(ordered_segments[segment_index])
            segment_index += 1
        #print "active segments before", active_segments
        below = above = 0
        for segment in list(active_segments):
            #print "interpolating", segment
            (x0, y0, x1, y1) = segment
            if x > x1:
                active_segments.remove(segment)
                #print "    segment removed"
            else:
                assert x0 <= x and x <= x1
                dx = x1 - x0
                #print "    dx", dx
                if dx > epsilon:
                    lmda = (x1 - x) * (1.0 / dx)
                    assert abs(x1 - dx * lmda - x) < epsilon
                    # interpolated y at x
                    yy = y1 - (y1 - y0) * lmda
                    #print "    interpolated y", yy, y
                    if yy < y:
                        #print "    below"
                        below += 1
                    else:
                        #print "    above"
                        above += 1
        #print "total below", below, "above", above
        if below % 2 == 1 and above % 2 == 1:
            index_to_point[i] = (x,y)
    return index_to_point

def test0():
    path = [[0,0], [0,1], [1,1], [1,0]]
    points = [((i-2)*0.2, (j-2)*0.25) for i in range(10) for j in range(10)]
    #points = [(0.5, 0.5)]
    inside = inside_lasso(points, path)
    print "inside unit square", inside
    for (i, (x,y)) in enumerate(points):
        if 0 < x < 1 and 0 < y < 1:
            assert i in inside, repr((x,y)) + " should be inside"
        else:
            if x < 0 or y < 0 or x > 1 or y > 1:
                assert i not in inside, repr((x,y)) + " should not be inside"
    print "assertions ok"

def test1():
    path = [[0,0], [1,1], [1,0]]
    points = [((i-2)*0.2, (j-2)*0.25) for i in range(10) for j in range(10)]
    #points = [(0.5, 0.5)]
    inside = inside_lasso(points, path)
    print "inside triangle", inside
    for (i, (x,y)) in enumerate(points):
        if 0 < x < 1 and 0 < y < 1 and x > y:
            assert i in inside, repr((x,y)) + " should be inside"
        else:
            if x < 0 or y < 0 or x > 1 or y > 1 or y > x:
                assert i not in inside, repr((x,y)) + " should not be inside"
    print "assertions ok"

if __name__ == "__main__":
    test0()
    test1()
