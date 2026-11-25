import torch
import isoext
import json
from model import SDF
import numpy as np
from utils import quat_apply

device, dtype = "cuda", torch.float32  # isoext needs float32 CUDA tensors


def local_box(name: str, a: np.ndarray):
    """Local-frame AABB (lo, hi) of {f < 0} for each primitive. `a` = |params|."""
    z = np.zeros(3)
    if name == "sdBox":
        return -a[:3], a[:3]
    if name == "sdCappedCylinder":            # params: r, h
        r, h = a
        return np.array([-r, -r, -h]), np.array([r, r, h])
    if name == "sdCappedCone":                # params: h, r1, r2
        h, r1, r2 = a
        R = max(r1, r2)
        return np.array([-R, -R, -h]), np.array([R, R, h])
    if name == "sdSegment":                   # params: h, runs 0..|h| along y
        return z, np.array([0.0, a[0], 0.0])
    if name == "sdPoint":
        return z, z
    if name == "sdRingSegment":               # params: theta, r; arc in xy plane
        theta, r = a
        half = min(theta, np.pi) / 2
        xe = r * (1.0 if half >= np.pi / 2 else np.sin(half))
        return np.array([-xe, r * np.cos(half), 0.0]), np.array([xe, r, 0.0])
    raise ValueError(f"no bounding box rule for {name}")  # fail loudly, never silently


@torch.no_grad()
def analytic_aabb(sdf):
    los, his, pad = [], [], 0.0
    for i, node in enumerate(sdf.nodes):
        a = node.params.flatten().abs().double().cpu().numpy()
        lo, hi = local_box(node.func.__name__, a)
        m = max(node.inflate.item(), 0.0)
        lo, hi = lo - m, hi + m

        # R from the project's own quat_apply: rows of the result are R @ e_i
        basis = torch.eye(3, device=node.rotation.device, dtype=node.rotation.dtype)[None]
        R = quat_apply(node.rotation, basis)[0].T.double().cpu().numpy()
        Rinv = np.linalg.inv(R)
        t = node.translation.flatten().double().cpu().numpy()

        c = Rinv @ ((lo + hi) / 2 + t)
        h = np.abs(Rinv) @ ((hi - lo) / 2)
        los.append(c - h)
        his.append(c + h)

        if i > 0 and node.sign.item() > 0:    # only smooth *unions* can grow the region
            pad += abs(node.smoothing.item()) / 4

    return np.min(los, 0) - pad, np.max(his, 0) + pad

@torch.no_grad()
def fit_bounds(sdf, probe_half=32.0, probe_res=64, pad=0.1, device="cuda"):
    """Return (aabb_min, aabb_max) enclosing the sdf < 0 region, as a cube."""
    g = isoext.UniformGrid(
        [probe_res] * 3,
        aabb_min=[-probe_half] * 3,
        aabb_max=[probe_half] * 3,
    )
    pts = g.get_points().reshape(-1, 3)
    vals = sdf(pts.unsqueeze(0)).squeeze(0)
    inside = pts[vals < 0]
    if inside.numel() == 0:
        raise ValueError("SDF has no interior within the probe box")

    lo, hi = inside.min(0).values, inside.max(0).values
    center = (lo + hi) / 2
    half = ((hi - lo).max() / 2) * (1 + pad) + 2 * (2 * probe_half / probe_res)
    # the last term pads by two probe cells so thin features aren't clipped
    if half >= probe_half:
        print("warning: interior touches the probe box; increase probe_half")
    return (center - half).tolist(), (center + half).tolist()


def as_field(sdf):
    """Wrap a batched SDF as a plain (N, 3) -> (N,) function."""
    def field(p: torch.Tensor) -> torch.Tensor:
        return sdf(p.unsqueeze(0)).squeeze(0)   # (1,N,3) -> (1,N) -> (N,)
    return field


def sdf_normals(field, pts: torch.Tensor) -> torch.Tensor:
    pts = pts.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        (g,) = torch.autograd.grad(field(pts).sum(), pts)
    g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.nn.functional.normalize(g, dim=-1).float().contiguous()


def reconstruct(sdf, grid, level: float = 0.0):
    field = as_field(sdf)

    # 1. Sample the SDF on the grid (no autograd needed here)
    pts = grid.get_points()
    with torch.no_grad():
        values = field(pts.reshape(-1, 3)).reshape(pts.shape[:-1])
    grid.set_values(values)

    # 2. Edge/surface crossings, then exact normals from the composite SDF
    its = isoext.get_intersection(grid, level=level)
    its.set_normals(sdf_normals(field, its.get_points()))

    # 3. Dual contouring using those Hermite data
    return isoext.dual_contouring(grid, level=level, intersection=its)

def touches_boundary(v: torch.Tensor, level: float) -> bool:
    faces = [v[0], v[-1], v[:, 0], v[:, -1], v[:, :, 0], v[:, :, -1]]
    return any(bool((f < level).any()) for f in faces)


def reconstruct_fitted(sdf, res=128, level=0.0, pad_cells=2, grow=1.25, max_grow=10):
    field = as_field(sdf)                      # from the earlier example
    lo, hi = analytic_aabb(sdf)
    center = (lo + hi) / 2
    half = float((hi - lo).max() / 2)          # cubic box -> isotropic cells

    for attempt in range(max_grow + 1):
        # leave `pad_cells` empty cells on each side so the surface stays interior
        hp = half * res / (res - 2 * pad_cells)
        grid = isoext.UniformGrid(
            [res] * 3,
            aabb_min=(center - hp).tolist(),
            aabb_max=(center + hp).tolist(),
        )
        pts = grid.get_points()
        with torch.no_grad():
            values = field(pts.reshape(-1, 3)).reshape(pts.shape[:-1])

        if not touches_boundary(values, level):
            break
        half *= grow                           # clipped: enlarge and resample
    else:
        raise RuntimeError("surface still touches the grid boundary after growing")

    grid.set_values(values)
    its = isoext.get_intersection(grid, level=level)
    its.set_normals(sdf_normals(field, its.get_points()))
    return isoext.dual_contouring(grid, level=level, intersection=its)

def recreate_obj_file(filepath: str, resolution: int = 128):
    with open(filepath) as f:
        sdf = SDF.deserialize(json.load(f), device, dtype)

    try: 
        #aabb_min, aabb_max = fit_bounds(sdf)
         vertices, faces = reconstruct_fitted(sdf)
    except RuntimeError, ValueError:
        #print("Potentially unbound model discovered, skipping")
        #aabb_min, aabb_max = [-6.0, -6.0, -6.0], [6.0, -6.0, 6.0],
        print("Error detected, skipping")

    #grid = isoext.UniformGrid([resolution, resolution, resolution], aabb_min=aabb_min, aabb_max=aabb_max)
    #vertices, faces = reconstruct(sdf, grid)
    isoext.write_obj(filepath + ".obj", vertices, faces)





#if __name__ == "__main__":