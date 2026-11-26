import sys
import numpy as np
sys.path.insert(0, "gp")
sys.path.insert(0, "utils")
import gp


def _cone_terms(self, position):
    s = (position[0] - self.center[0], position[1] - self.center[1],
         position[2] - self.center[2])
    g = gp.compute_dot_product(s, self.axis_dir)
    slen = gp.compute_length_vectorized(s)
    f = np.sqrt(np.maximum(slen * slen - g * g, 0.0))
    da = np.cos(self.angle) * f
    db = -np.sin(self.angle) * g
    idx = np.logical_and(g < 0.0, (da - db) < 0.0)
    return da, db, slen, idx


def _distance(self, position):
    da, db, slen, idx = _cone_terms(self, position)
    res = np.abs(da + db)
    res[idx] = slen[idx]
    return res


def _signed_distance(self, position):
    da, db, slen, idx = _cone_terms(self, position)
    res = -(da + db)
    res[idx] = -slen[idx]
    return res


gp.ConePrimitive.distance = _distance
gp.ConePrimitive.signed_distance = _signed_distance

if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) != 8:
        gp.usage(sys.argv[0])
        sys.exit(1)
    gp.main(a[0], a[1], a[2], a[3], popsize=int(a[4]), maxgen=int(a[5]),
            mutationrate=float(a[6]), breedingrate=float(a[7]))