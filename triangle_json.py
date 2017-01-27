
# TEMPORARY HACK

from contourist import pentatopes

text = open("misc/triangles_test.template.html").read()

#G = pentatopes.test0()
#Mt = G.collect_morph_triangles()
#json = Mt.to_json()

import numpy as np
from numpy.linalg import norm

def f(x, y, z, t=0):
    # sphere
    return norm([x, y, z])

def f2(x, y, z, t):
    return norm([x, y, z, t])

def g0(x, y, z, t=0):
    # cube
    result = np.max(np.abs([x, y, z]))
    #print x,y,z,result
    return result

def g2(x, y, z, t=0):
    # torus-like
    alpha = norm([x, y])
    return 3 * norm([1 - alpha, z])

def bar(x, y, z, t=0):
    return 3 * norm([x, z])

def fg(x, y, z, t):
    # shape morph (?)
    return t * bar(x,y,z) + (1 - t)*g2(x,y,z)

G = pentatopes.MorphingIsoSurfaces([-2,-2,-2,0], [2,2,2,1], [0.2]*4, fg, 1.2, [])
print "looking for endpoints"
G.search_for_endpoints()
print "calculating morph from", len(G.grid_endpoints), "endpoint pairs"
#Mt = G.collect_morph_triangles()
json = G.to_json()
print "dumping json"
#json = Mt.to_json()
out_filename = "misc/triangles.json"
open(out_filename, "w").write(json)
print "output to", out_filename
