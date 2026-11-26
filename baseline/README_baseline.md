This folder reproduces Fayolle & Pasko's method: https://github.com/fayolle/csg_recovery_py

`gp/`, `segmentation/` and `utils/` are his code (`gp/gp.py` is unmodified).
`obj_to_xyzn.py`, `run_gp.py` and `extract_fayolle_meshes.py` were written for this project.

## Pre-check
- Install the Python packages: `pip install -r requirements_baseline.txt`
  (his own `requirements.txt` lists only numpy).
- Put the target `.obj` files in `../datasets/shapes`.

## Shortcut: skip the build

The `.xyzn` files are in `data/` and the `.fit` files are in `out/ransac_segps/`.
With those, you can start at step 3 (genetic programming) with no build. First create
the output folders:

```bash
mkdir -p out/csg_trees out/logs out/obj
```

- `gp/gp.py` is Fayolle's, unmodified. It reads a `.fit` and an `.xyzn` and writes
  `{shape}_expression.txt` and `{shape}_primitives.txt`, which together define the CSG tree.
- Shapes whose `.fit` contains cones (dog, hand, sofa) crash in his unmodified code
  because of an indexing bug in `ConePrimitive`. `run_gp.py` applies a runtime fix and
  then calls his `gp.main`, so use it for those shapes. Rod and pot have no cones.
- `extract_fayolle_meshes.py` then turns the expression, primitives and `.fit` files
  into meshes in `out/obj/`.

## Steps
### 1. Build segmentation program
```bash
mkdir -p bin data out/ransac_segps out/csg_trees out/logs out/obj
cd segmentation
mkdir -p build_linux && cd build_linux
cmake .. && make
cp ransac_command ../../bin/     # the binary name may differ; check what make produced
cd ../..
```

### 2. Convert meshes to points, then segment
```bash
python obj_to_xyzn.py

for s in dog hand pot rod sofa; do
  bin/ransac_command data/$s.xyzn out/ransac_segps/$s-segmented.segps > out/logs/ransac_$s.txt
done
```

### 3. Genetic programming
Rod and pot (no cones) use his `gp.py`:

```bash
for s in rod pot; do
  nohup python -u gp/gp.py out/ransac_segps/$s-segmented.fit data/$s.xyzn \
    out/csg_trees/${s}_expression.txt out/csg_trees/${s}_primitives.txt \
    150 3000 0.3 0.3 > out/logs/$s.log 2>&1 &
done
```

Dog, hand and sofa (cones) use `run_gp.py`:

```bash
for s in dog hand sofa; do
  nohup python -u run_gp.py out/ransac_segps/$s-segmented.fit data/$s.xyzn \
    out/csg_trees/${s}_expression.txt out/csg_trees/${s}_primitives.txt \
    150 3000 0.3 0.3 > out/logs/$s.log 2>&1 &
done
```

The runs are in the background so wait until they finish before proceeding to step 4.
They write their expression files only at the end, so check `out/csg_trees/` to know
when they're done. Can also run this to check them:
```bash
for s in rod pot dog hand sofa; do
  echo -n "$s: "; grep -E '^[0-9.e+-]+$' out/logs/$s.log | awk 'END{print NR" iters, best "$1}'
done
```

### 4. Reconstruct meshes
```bash
python extract_fayolle_meshes.py
```

Writes `out/obj/Fayolle_{shape}.obj`. Shapes that haven't finished are skipped.

## gp.py vs run_gp.py

`gp/gp.py` is Fayolle's, unmodified. When a `.fit` file contains cone primitives, his
`ConePrimitive.distance` and `signed_distance` crash because they assign a full-size
array into a boolean-masked slice (`res[idx] = -np.sqrt(sqrs)`). Shapes without cones
(rod, pot) are unaffected.

`run_gp.py` is a wrapper, run in place of `gp.py` for shapes whose `.fit` has cones
(dog, hand, sofa). It replaces those two cone methods at runtime with a corrected
version (`res[idx] = -slen[idx]`, the same formula applied to the selected points only)
and then calls his `gp.main` with the same arguments, so the rest of the algorithm is
his code. `extract_fayolle_meshes.py` imports it so reconstruction uses the same cone
math. To check whether a shape needs it: `grep -l "^cone" out/ransac_segps/*.fit`.
