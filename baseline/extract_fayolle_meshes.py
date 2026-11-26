import os
import sys
import collections
import numpy as np
import trimesh
from skimage.measure import marching_cubes

# Import Fayolle's exact primitive definitions and expression parser
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gp"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))

import gp
import run_gp  # patches ConePrimitive.distance/signed_distance (his gp.py stays unchanged)
import tree_from_expression as tfe


def eval_csg_node(node, prim_map, pts):
    """Evaluate Fayolle's CSG tree (positive inside, negative outside)."""
    label = node.label
    if label in prim_map:
        return prim_map[label].signed_distance(pts)
    elif label == "negation":
        return -eval_csg_node(node.left, prim_map, pts)
    elif label == "union":
        return np.maximum(
            eval_csg_node(node.left, prim_map, pts),
            eval_csg_node(node.right, prim_map, pts),
        )
    elif label == "intersection":
        return np.minimum(
            eval_csg_node(node.left, prim_map, pts),
            eval_csg_node(node.right, prim_map, pts),
        )
    elif label == "subtraction":
        return np.minimum(
            eval_csg_node(node.left, prim_map, pts),
            -eval_csg_node(node.right, prim_map, pts),
        )
    else:
        raise ValueError(f"Unknown CSG node label: {label}")


def reconstruct_shape(shape, grid_res=128, pad_ratio=0.10):
    fit_path = f"out/ransac_segps/{shape}-segmented.fit"
    xyzn_path = f"data/{shape}.xyzn"
    exp_path = f"out/csg_trees/{shape}_expression.txt"
    prim_list_path = f"out/csg_trees/{shape}_primitives.txt"
    out_obj_path = f"out/obj/Fayolle_{shape}.obj"

    # Load primitives and map names (e.g., 'plane0', 'cylinder1')
    primitives = gp.read_fit(fit_path)
    prim_map = {}
    for idx, p in enumerate(primitives):
        prim_map[f"{p.identifier()}{idx}"] = p

    # Parse expression into binary tree using Fayolle's parser
    expr_str = tfe.read_expression_from_file(exp_path)
    tfe.PRIMITIVES = tfe.read_primitives_list(prim_list_path)
    preorder = tfe.construct_tree(expr_str)
    tree = tfe.build_tree(collections.deque(preorder))

    # Compute bounding box from input point cloud + padding
    # paddings ensures surface isn't clipped right at the edges
    pts_cloud = np.loadtxt(xyzn_path)[:, :3]
    bmin = pts_cloud.min(axis=0)
    bmax = pts_cloud.max(axis=0)
    extent = bmax - bmin
    pad = np.maximum(extent * pad_ratio, 0.05)
    gmin = bmin - pad
    gmax = bmax + pad

    # Build 3D regular grid and evaluate CSG SDF
    xs = np.linspace(gmin[0], gmax[0], grid_res)
    ys = np.linspace(gmin[1], gmax[1], grid_res)
    zs = np.linspace(gmin[2], gmax[2], grid_res)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    sdf = eval_csg_node(tree, prim_map, (X, Y, Z))

    # Clip to bounding box (Fayolle's convention: positive inside box)
    # caps infinite primitives at grid walls
    bbox_sdf = np.minimum.reduce([
        X - gmin[0], gmax[0] - X,
        Y - gmin[1], gmax[1] - Y,
        Z - gmin[2], gmax[2] - Z,
    ])
    sdf = np.minimum(sdf, bbox_sdf)

    # Extract zero-isosurface via Marching Cubes
    if sdf.min() >= 0 or sdf.max() <= 0:
        print(f"[WARNING] {shape}: SDF does not cross zero (min={sdf.min():.4f}, max={sdf.max():.4f})")
        return

    spacing = (
        (gmax[0] - gmin[0]) / (grid_res - 1),
        (gmax[1] - gmin[1]) / (grid_res - 1),
        (gmax[2] - gmin[2]) / (grid_res - 1),
    )
    verts, faces, normals, _ = marching_cubes(sdf, level=0.0, spacing=spacing)
    verts += gmin

    # Reverse triangle's rotational direction so normals point outward (since sdf > 0 inside)
    faces = faces[:, ::-1]

    os.makedirs(os.path.dirname(out_obj_path), exist_ok=True)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh.export(out_obj_path)
    print(f"[OK] {shape} -> {out_obj_path} ({len(mesh.vertices)} verts, {len(mesh.faces)} faces)")


if __name__ == "__main__":
    for s in ["dog", "hand", "pot", "rod", "sofa"]:
        print(f"Reconstructing {s}...")
        try:
            reconstruct_shape(s, grid_res=128)
        except FileNotFoundError as e:
            print(f"  skipped, not finished yet: {e.filename}")
