import glob
import os
import numpy as np
import trimesh

OBJ_FOLDER = "../datasets/shapes"
OUT_FOLDER = "data"          # rename your old data folder first (see below)


def smooth_sample(mesh, n=25000, crease_deg=45):
    """Sample n points with angle-weighted vertex normals, averaged only
    across faces within crease_deg of each other (keeps sharp edges)."""
    fn, F, face_ang = mesh.face_normals, mesh.faces, mesh.face_angles
    vf = mesh.vertex_faces                      # (V, k) face ids, padded with -1
    cos_t = np.cos(np.radians(crease_deg))
    corner = np.zeros((len(F), 3, 3))           # one normal per corner of each face
    for c in range(3):
        v = F[:, c]
        nb = vf[v]                              # faces around this corner's vertex
        valid = nb >= 0
        idx = np.where(valid, nb, 0)
        nbn = fn[idx]                           # (F, k, 3)
        # corner angle of each neighbouring face at this vertex
        corner_ang = (face_ang[idx] * (F[idx] == v[:, None, None])).sum(-1)
        w = corner_ang * valid
        keep = (np.einsum('fkj,fj->fk', nbn, fn) >= cos_t) * w
        s = np.einsum('fk,fkj->fj', keep, nbn)
        corner[:, c] = s / np.maximum(np.linalg.norm(s, axis=1, keepdims=True), 1e-12)
    pts, fidx = mesh.sample(n, return_index=True)
    bary = trimesh.triangles.points_to_barycentric(mesh.triangles[fidx], pts)
    nrm = np.einsum('ij,ijk->ik', bary, corner[fidx])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    return pts, nrm


os.makedirs(OUT_FOLDER, exist_ok=True)
obj_files = sorted(glob.glob(os.path.join(OBJ_FOLDER, "*.obj")))

if not obj_files:
    print(f"ERROR: No .obj files found in: {os.path.abspath(OBJ_FOLDER)}")
else:
    for obj_path in obj_files:
        name = os.path.splitext(os.path.basename(obj_path))[0]
        mesh = trimesh.load(obj_path, force="mesh")
        mesh.merge_vertices()
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
        # 25000 sample points best for RANSAC and CPU runtime
        points, normals = smooth_sample(mesh, 25000)
        # place each sample point's normal direction beside the points' coordinates
        xyzn = np.hstack((points, normals))
        out_path = os.path.join(OUT_FOLDER, f"{name}.xyzn")
        np.savetxt(out_path, xyzn, fmt="%.6f")
        print(f"Saved -> {os.path.abspath(out_path)}")