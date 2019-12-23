
import numpy as np
import scipy.optimize

def inside(point, points, test=False):
    "test whether the point is inside the convex closure of points."
    points = np.array(points)
    point = np.array(point)
    (n, d) = points.shape
    points_T = points.transpose()
    n_ones = np.ones((n,))
    A = np.vstack([points_T, -points_T, -n_ones, n_ones])
    b = np.hstack([point, -point, [-1, 1]])
    c = np.ones((n,))
    r = scipy.optimize.linprog(c, A, b)
    if test:
        r.A = A
        r.b = b
        r.c = c
        return r
    return r.success

def l1_fit(U, v, strict=True):
    """
    Find a least absolute error solution (m, k) to U * m + k =approx= v 
    """
    U = np.array(U)
    v = np.array(v)
    # n is the number of samples
    n = len(v)
    s = U.shape
    assert len(s) == 2
    assert s[0] == n
    # d is the number of dimensions
    d = s[1]
    # find an upper bound on the errors
    m_test = np.ones((d,1))
    error_upper_bounds = np.abs(U.dot(m_test)) + 1
    #print "upper bounds", error_upper_bounds
    error_bounds = [(0, e) for e in error_upper_bounds]
    I = np.identity(n)
    n1 = np.ones((n,1))
    A = np.vstack([
            np.hstack([-I, U, n1]),
            np.hstack([-I, -U, -n1])
        ])
    c = np.hstack([np.ones(n), np.zeros(d+1)])
    b = np.hstack([v, -v])
    bounds = [(0, None)] * n + [(None, None)] * (d+1)
    bounds = error_bounds + [(None, None)] * (d+1)
    options = {"maxiter": 10000}
    r = scipy.optimize.linprog(c, A, b, bounds=bounds, options=options)
    #print r.message
    if r.success:
        x = r.x
        m = x[n:n+d]
        k = x[n+d]
    else:
        # fit failed (due to a bug in linprog?)
        if strict:
            raise ValueError("fit failed: " + repr(r))
        #print "fit failed", error_upper_bounds
        x = error_upper_bounds
        m = m_test
        k = 0
    residuals = v - (np.dot(U, m) + k)
    result = {}
    result["U"] = U
    result["v"] = v
    result["m"] = m
    result["k"] = k
    result["r"] = r
    result["A"] = A
    result["b"] = b
    result["c"] = c
    result["bounds"] = bounds
    result["residuals"] = residuals
    result["errors"] = x[:n]
    return result

def l2_fit(x, y, strict=True):
    n = len(x)
    A = np.hstack([x, np.ones(n).T.reshape((n,1))])
    lstsq = np.linalg.lstsq(A, y)[0]
    m = lstsq[:-1]
    k = lstsq[-1]
    result = {}
    result["U"] = x
    result["v"] = y
    result["m"] = m
    result["k"] = k
    result["errors"] = np.abs(y - (np.dot(x, m) + k))
    return result

def check_flatten(result):
    m = fit["m"]
    points = result["points"]
    k = result["k"]
    fitting = result["fitting"]
    errors = np.abs(points.dot(m) + k - fitting).sum()
    terror = result["total_error"]
    assert np.allclose(errors, terror)

def flatten(points, fitter=l1_fit):
    "Use l1_fit to flatten points onto a heuristic plane, minimizing error in one of the dimensions."
    points = np.array(points)
    (n, d) = points.shape
    assert d > 1
    maxes = points.max(axis=0)
    mins = points.min(axis=0)
    diff = maxes - mins
    flattest_index = diff.argmin()
    v = points[:, flattest_index]
    U = np.hstack([points[:, i].reshape(n, 1) for i in range(d) if i!=flattest_index]).reshape(n, d-1)
    fit = fitter(U, v, strict=False)
    result = {}
    #result["fit"] = fit
    m = list(fit["m"])
    m.insert(flattest_index, 0)
    result["points"] = points
    result["m"] = m
    result["k"] = fit["k"]
    result["projection"] = U
    result["fitting"] = v
    result["flattest_index"] = flattest_index
    result["maxes"] = maxes
    result["mins"] = mins
    #result["total_error"] = fit["errors"].sum()
    result["errors"] = fit["errors"]
    result["extent"] = diff.max()
    return result

def nearly_flat(p0, p1, points, minimum_extent, minimum_ratio):
    """
    test whether line segement p0 to p1 are nearly flat inside points
    and should be merged into a single point.
    """
    p0 = np.array(p0)
    p1 = np.array(p1)
    segment_extent = np.abs(p0 - p1).max()
    if segment_extent < minimum_extent:
        # The line segment is too small: merge the points
        #print "collapse tiny segment"
        return True
    # fit the points to a plane
    fit_points = np.vstack([[p0, p1], points])
    try:
        fit = flatten(fit_points)
    except:
        #print "failed to flatten", fit_points
        raise
    projection = fit["projection"]
    p0_proj, p1_proj = projection[:2]
    points_proj = projection[2:]
    # only merge if p0, p1 lie inside the other points in plane projection.
    if inside(p0_proj, points_proj) and inside(p1_proj, points_proj):
        errors = fit["errors"]
        #print "both inside...", errors
        # if the max error is less than the min extent, then the points are flat-ish
        if errors.max() < minimum_extent:
            #print "plane fit errors are small: it's flat"
            return True
        #avg_error = errors.sum() * 1.0/len(fit_points)  # should use max or median or other?
        avg_error = errors.max()
        extent = fit["extent"]
        ratio = avg_error / extent
        if ratio < minimum_ratio:
            # if the errors don't vary much relative to the extent, it's flat
            #print "small errors relative to extent: it's flat", ratio, minimum_ratio, avg_error, extent
            return True
        #print "large errors relative to extent", ratio, minimum_ratio, avg_error, extent
    # default
    #print "default: not flat"
    return False
