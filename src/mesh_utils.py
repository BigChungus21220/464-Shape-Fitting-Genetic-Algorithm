from collections.abc import Sequence
import torch
import isoext as ix
from pathlib import Path

from model import SDF

def reconstruct_sdf(sdf: SDF, lower: Sequence[float], upper: Sequence[float], grid_resolution: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstructs a mesh from an SDF

    Args:
        sdf (SDF): A function implementing isoext.sdf.SDFProtocol
        lower (tuple[float, float, float]): Lower sampling corner
        upper (tuple[float, float, float]): Upper sampling corner
        grid_resolution (int): Grid resolution on each axis

    Returns:
        (vertices: (N, 3), faces: (M, 3)): Reconstructed mesh
    """
    device = torch.device('cuda')
    dtype = torch.float32
    sdf_ix = sdf.to(device, dtype)
    
    grid = ix.UniformGrid([grid_resolution, grid_resolution, grid_resolution], lower, upper)
    grid.set_values(sdf_ix(grid.get_points()))
    
    intersection = ix.get_intersection(grid)
    points = ix.sdf.project_to_surface(sdf_ix, intersection.get_points())
    intersection.set_points(points)
    
    intersection.set_normals(ix.sdf.get_sdf_normal(sdf_ix, points))
    
    return ix.dual_contouring(grid, intersection=intersection, clamp=False)


def resample_mesh(tris: torch.Tensor, tri_normals: torch.Tensor, num_samples: int):
    """Resamples points on the mesh surface uniformly by area

    Args:
        tris (torch.Tensor): (T, 3, 3) vertex positions
        tri_normals (torch.Tensor): (T, 3) triangle normals
    Returns:
        (torch.Tensor, torch.Tensor): resampled verts and normals
    """
    v0 = tris[:, 0]  # (T, 3)
    e1 = tris[:, 1] - v0
    e2 = tris[:, 2] - v0

    areas = 0.5 * torch.linalg.norm(torch.cross(e1, e2, dim=-1), dim=-1)  # (T,)
    idx = torch.multinomial(areas, num_samples=num_samples, replacement=True)

    r1 = torch.rand((num_samples, 1), device=tris.device, dtype=tris.dtype).sqrt()
    r2 = torch.rand((num_samples, 1), device=tris.device, dtype=tris.dtype)
    points = v0[idx] + r1 * (1.0 - r2) * e1[idx] + (r1 * r2) * e2[idx]

    return points, tri_normals[idx]