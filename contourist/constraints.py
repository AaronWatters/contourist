"""
Linear constraint logic.
"""

import numpy as np
from numpy import linalg
import itertools
from scipy.optimize import linprog

class ConstrainedLevelSet(object):

    def __init__(self, f, v, A, b):
        """
        Level set function which is >0 where f(p)>v and A p < b, negative elsewhere.
        """
        self.f = f
        self.v = v
        self.A = np.array(A)
        self.b = np.array(b)

    def __call__(self, *p):
        f_factor = self.f(*p) - self.v
        constraint_factor = (self.b - self.A.dot(p)).min()
        return min(constraint_factor, f_factor)

class Constraints(object):
    "logic relating to geometric constraint coef.dot(x) <= consts"

    def __init__(self, dimension=3):
        self.dimension = dimension
        self.coefs = []
        self.consts = []
        self.labels = []

    def add(self, coef, const, label=None):
        assert len(coef) == self.dimension
        self.coefs.append(coef)
        self.consts.append(const)
        self.labels.append(label)

    def add_points(self, points, point_inside, label=None):
        "Add constraint where points lie on the constraint plane and point_inside satisfies the constraint"
        d = self.dimension
        d1 = d + 1
        assert len(points) == d
        elements = [list(p) + [1] for p in points]
        a = np.array(elements + [1] * (d1))
        assert a.shape == (d1, d1)
        b = np.ones((d1,))
        soln = linalg.solve(a, b)
        coef = soln[:-1]
        const = soln[-1]
        test = coef.dot(point_inside)
        if test > const:
            coef = - coef
            const = - const
        return self.add(coef, const, label)

    def zero_level_function(self, f, v):
        "return Level set function which is >0 where f(p)>v and A p < b, negative elsewhere."
        return ConstrainedLevelSet(f, v, self.coefs, self.consts)

    def feasible_vertices_iter(self):
        coefs = np.array(self.coefs)
        consts = np.array(self.consts)
        for indices in itertools.combinations(range(len(coefs)), self.dimension):
            indices = list(indices)
            a = coefs[indices]
            b = consts[indices]
            try:
                vertex = linalg.solve(a, b)
            except linalg.LinAlgError:
                pass
            else:
                test = consts - coefs.dot(vertex)
                #print (indices, vertex, test)
                if np.all(test >= 0):
                    yield (frozenset(indices), vertex, test)
                
    def feasible_vertices(self):
        return list(self.feasible_vertices_iter())
    
    def feasible_faces(self):
        "Return points and faces as sequences of points in order around the perimeter of each face."
        points = []
        faces_indices = []
        labels = []
        constraint_index_triples_to_point_index = {}
        constraint_to_triples = {}
        for (constraints, point, test) in self.feasible_vertices():
            point_index = len(points)
            points.append(point)
            constraint_index_triples_to_point_index[constraints] = point_index
            for constraint in constraints:
                triples = constraint_to_triples.setdefault(constraint, set())
                triples.add(constraints)
        for face_constraint in constraint_to_triples:
            labels.append(self.labels[face_constraint])
            triples = constraint_to_triples[face_constraint]
            triple = triples.pop()
            point_index = constraint_index_triples_to_point_index[triple]
            face_indices = [point_index]
            # remove triple one constraint different until no more triples (or error)
            while triples:
                next_triple = None
                for triple0 in triples:
                    diff = triple0 - triple
                    if len(diff) == 1:
                        next_triple = triple0
                        break
                assert next_triple is not None, "no suitable triple " + repr(triples)
                triple = next_triple
                triples.remove(triple)
                point_index = constraint_index_triples_to_point_index[triple]
                face_indices.append(point_index)
            faces_indices.append(face_indices)
        return (points, faces_indices, labels)
    
    def labelled_faces(self):
        (points, faces_indices, labels) = self.feasible_faces()
        label_to_points = {}
        for (label, indices) in zip(labels, faces_indices):
            if label is not None:
                face_points = [points[i] for i in indices]
                label_to_points[label] = np.array(face_points)
        return label_to_points
    
    def triangulation(self):
        (points, faces_indices, labels) = self.feasible_faces()
        triples = []
        for face_indices in faces_indices:
            if len(face_indices) > 2:
                (a, b) = face_indices[:2]
                for c in face_indices[2:]:
                    triples.append((a, b, c))
                    b = c
        return (points, triples)
    
    def optimize_gradient(self, gradient):
        A_ub = np.array(self.coefs)
        b_ub = np.array(self.consts)
        bounds = [(None, None)] * 3   # no automatic boundaries
        solution = linprog(gradient, A_ub, b_ub)
        return solution.x
