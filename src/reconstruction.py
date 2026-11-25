from collections.abc import Sequence
import torch
import isoext as ix

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