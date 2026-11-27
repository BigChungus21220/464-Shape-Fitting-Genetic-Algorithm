# evaluation metrics: chamfer distance and volumetric IoU between a target mesh and an SDF

from collections.abc import Sequence
import numpy as np
import torch
import trimesh

from mesh_utils import reconstruct_sdf, resample_mesh
from model import SDF


def winding_number(points: torch.Tensor, tris: torch.Tensor, point_chunk: int = 1024, tri_chunk: int = 8192) -> torch.Tensor:
    """Generalized winding number of each point w.r.t. a triangle soup (Jacobson et al. 2013).
    Robust to small holes / non-watertight meshes; ~1 inside, ~0 outside.

    Args:
        points (torch.Tensor): (N, 3)
        tris (torch.Tensor): (T, 3, 3)

    Returns:
        torch.Tensor: (N,)
    """
    out = torch.zeros(points.shape[0], device=points.device, dtype=points.dtype)
    for i in range(0, points.shape[0], point_chunk):
        p = points[i:i + point_chunk, None, None, :] # (n, 1, 1, 3)
        total = torch.zeros(p.shape[0], device=points.device, dtype=points.dtype)
        for j in range(0, tris.shape[0], tri_chunk):
            v = tris[None, j:j + tri_chunk] - p # (n, t, 3, 3)
            a, b, c = v.unbind(2)
            la, lb, lc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
            det = (a*torch.cross(b, c, dim=-1)).sum(-1)
            denom = la*lb*lc + (a*b).sum(-1)*lc + (b*c).sum(-1)*la + (c*a).sum(-1)*lb
            total += (2*torch.atan2(det, denom)).sum(-1) # solid angle per triangle
        out[i:i + point_chunk] = total/(4*torch.pi)
    return out


def chamfer_distance(a: torch.Tensor, b: torch.Tensor, chunk: int = 2048) -> float:
    """Symmetric chamfer distance: mean over both directions of the (unsquared) nearest-neighbour distance.

    Args:
        a (torch.Tensor): (N, 3)
        b (torch.Tensor): (M, 3)
    """
    def one_way(x: torch.Tensor, y: torch.Tensor):
        return torch.cat([torch.cdist(x[i:i + chunk], y).amin(dim=1) for i in range(0, x.shape[0], chunk)]).mean()
    return 0.5*(one_way(a, b) + one_way(b, a)).item()


class ShapeMetrics:
    """Precomputes target samples once, then scores SDFs against them."""

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        lower: Sequence[float],
        upper: Sequence[float],
        surface_samples: int = 5000,
        volume_samples: int = 50000,
        resolution: int = 64,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            mesh (trimesh.Trimesh): target mesh
            lower, upper: bounds of the sampling box (target bounds plus padding)
            surface_samples (int): points sampled on each surface for chamfer distance
            volume_samples (int): uniform points in the bounding box for IoU
            resolution (int): grid resolution used to reconstruct the SDF surface for chamfer distance
        """
        # some targets (e.g. sofa) are double-sided: every face appears twice with opposite
        # orientation, so winding numbers cancel. Drop duplicates and make orientation consistent.
        _, keep = np.unique(np.sort(mesh.faces, axis=1), axis=0, return_index=True)
        if len(keep) < len(mesh.faces):
            mesh = trimesh.Trimesh(mesh.vertices, mesh.faces[np.sort(keep)], process=True)
            trimesh.repair.fix_normals(mesh, multibody=True)
        tris = torch.tensor(mesh.triangles, device=device, dtype=dtype)

        self.lower, self.upper = list(lower), list(upper)
        self.surface_samples = surface_samples
        self.resolution = resolution
        
        with torch.random.fork_rng():
            dummy_normals = torch.zeros(tris.shape[0], 3, device=device, dtype=dtype)
            self.target_surface, _ = resample_mesh(tris, dummy_normals, surface_samples)

            lo = torch.tensor(self.lower, device=device, dtype=dtype)
            hi = torch.tensor(self.upper, device=device, dtype=dtype)
            self.volume_points = lo + (hi - lo)*torch.rand(volume_samples, 3, device=device, dtype=dtype)
        with torch.no_grad():
            self.target_inside = winding_number(self.volume_points, tris).abs() > 0.5

    @torch.no_grad()
    def iou(self, sdf: SDF) -> float:
        sdf = sdf.to(self.volume_points.device, self.volume_points.dtype)
        pred_inside = sdf(self.volume_points) < 0
        inter = (pred_inside & self.target_inside).sum().item()
        union = (pred_inside | self.target_inside).sum().item()
        return inter/union if union > 0 else 1.0

    @torch.no_grad()
    def chamfer(self, sdf: SDF) -> float:
        verts, faces = reconstruct_sdf(sdf, self.lower, self.upper, self.resolution)
        if faces.numel() == 0:
            return float("inf") # no surface inside the box
        verts = verts.to(self.target_surface.device, self.target_surface.dtype)
        tris = verts[faces.long()]
        areas = torch.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0], dim=-1).norm(dim=-1)
        if not torch.isfinite(areas).all() or areas.sum() <= 0:
            return float("inf")
        dummy_normals = torch.zeros(tris.shape[0], 3, device=tris.device, dtype=tris.dtype)
        pred_surface, _ = resample_mesh(tris, dummy_normals, self.surface_samples)
        return chamfer_distance(pred_surface, self.target_surface)

    def __call__(self, sdf: SDF) -> dict[str, float]:
        with torch.random.fork_rng():
            return {"chamfer": self.chamfer(sdf), "iou": self.iou(sdf)}
