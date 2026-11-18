A genetic algorithm for shape fitting

## Current strategy

### Fitness Function:

Given a pointcloud with normals, points `p_i`, normals `n_i`

`fitness = sum(sdf(p_i)^2 + k*(1 - dot(sdf_normal(p_i), normal(n_i)))^2) + num_objects`

Where the sdf is defined recusively by `sdf_{i+1} = smooth_union_or_difference(sdf_i, k_{i+1}, X_{i+1})`

and `X_i` is a rounded box sdf, defined by a center point (3d), size (3d), and edge radii (12d)

and `k_i` (1d) is the smoothness of the boolean operation.


### Genes:

Each individual has two gene types
 - genes representing cuboids (18d for the parameters + 1 bool for union or difference)
 - genes representing intersection smoothness (1d)

Mutation:
 - add a random new cube
 - add a random amount to a float parameter

Crossover:
 - smoothness: choose one of the parents or the average
 - cube: random choice between parent cubes


### Training:
 - use keep best + weighted choice




