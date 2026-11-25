# model loading

import torch


def unique_indices(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the indices of unique elements of x (no guaruntee on order of occurence) and the inverse mapping from x to the index list
    """
    unique_vals, inverse = torch.unique(x, return_inverse=True)
    rep_idx = torch.empty(unique_vals.shape[0], dtype=torch.long, device=x.device)
    rep_idx[inverse] = torch.arange(x.shape[0], device=x.device)
    return rep_idx, inverse

def norm(x: torch.Tensor) -> torch.Tensor:
    """
    (...,N) -> (...,1)
    """
    return torch.linalg.vector_norm(x, dim=-1, keepdim=True)

def norm2(x: torch.Tensor):
    """
    (...,N) -> (...,1)
    """
    return torch.sum(torch.square(x), dim=-1, keepdim=True)

def dot(x: torch.Tensor, y: torch.Tensor):
    """
    (...,N), (...,N) -> (...,1)
    """
    return torch.sum(x*y, dim=-1, keepdim=True)

def cat(*args: torch.Tensor):
    """
    (...,Ni)... -> (...,sum(Ni))
    """
    return torch.cat(args, dim=-1)

def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    Rotate 3D vectors by unit quaternions. In programming convention (x,y,z,w).
    
    Args:
        quat (Tensor): (..., 4)
        vec (Tensor):  (..., 3)
    Returns:
        Rotated vectors of shape (..., 3).
    """
    
    xyz = quat[..., :3]
    w = quat[..., 3:]
    
    t = 2*torch.linalg.cross(xyz, vec, dim=-1)
    return vec + w*t + torch.linalg.cross(xyz, t, dim=-1)

def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Multiplies unit quaternions. In programming convention (x,y,z,w).
    
    Args:
        q1 (Tensor): (..., 4)
        q2 (Tensor): (..., 4)
    Returns:
        Rotated vectors of shape (..., 3).
    """
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack((x, y, z, w), dim=-1)

def quat_random(batch_size: int, max_angle: float, device: torch.device | str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Generates random quaternions (uniform in angle applied)
     -> (B,1,4)
    """
    axis = torch.randn((batch_size, 3), device=device, dtype=dtype)
    axis = torch.nn.functional.normalize(axis, p=2, dim=-1)

    half_theta = 0.5*max_angle*torch.rand((batch_size, 1), device=device, dtype=dtype)

    q = cat(axis*torch.sin(half_theta), torch.cos(half_theta))

    return q.unsqueeze(1)