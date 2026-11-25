import torch
import isoext
import json
from model import SDF

device, dtype = "cuda", torch.float32  # isoext needs float32 CUDA tensors

@torch.no_grad()
def fit_bounds(sdf, probe_half=8.0, probe_res=64, pad=0.1, device="cuda"):
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


def recreate_obj_file(filepath: str, resolution: int = 128):
    with open(filepath) as f:
        sdf = SDF.deserialize(json.load(f), device, dtype)

    try: 
        aabb_min, aabb_max = fit_bounds(sdf)
        grid = isoext.UniformGrid([resolution, resolution, resolution], aabb_min=aabb_min, aabb_max=aabb_max)
        vertices, faces = reconstruct(sdf, grid)
        isoext.write_obj(filepath + ".obj", vertices, faces)
    except ValueError:
        print("Potentially unbound model discovered, skipping")




#if __name__ == "__main__":