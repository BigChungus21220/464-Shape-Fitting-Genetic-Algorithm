# sdf definitions and operators

from collections.abc import Callable
import torch

from utils import norm, norm2, dot, cat

def estimateNormal(sdf: Callable[[torch.Tensor], torch.Tensor], p: torch.Tensor, eps: float = 1e-5):
    """Estimates the normal of a scalar field

    Args:
        sdf (Callable[[torch.Tensor], torch.Tensor]): (B, V, 3) -> (B, V)
        p (torch.Tensor): (B, V, 3)
        eps (float, optional): measurment step. Defaults to 1e-5.

    Returns:
        torch.Tensor: (B, V, 3) normals (unit length)
    """
    B, V, _ = p.shape
    
    offsets = torch.tensor([
        [ eps, 0.0, 0.0],
        [-eps, 0.0, 0.0],
        [0.0,  eps, 0.0],
        [0.0, -eps, 0.0],
        [0.0, 0.0,  eps],
        [0.0, 0.0, -eps],
    ], device=p.device, dtype=p.dtype) # (6, 3)
    
    # (B, 6, V, 3) -> (B, 6 * V, 3)
    pts = (p[:, None, :, :] + offsets[None, :, None, :]).reshape(B, 6 * V, 3)

    # (B, 6 * V) -> (B, 6, V)
    vals = sdf(pts).view(B, 6, V)

    # Central differences: (+offset) - (-offset) -> (B, 3, V) -> (B, V, 3)
    grad = (vals[:, 0::2] - vals[:, 1::2]).transpose(1, 2)

    return grad / norm(grad)


def sdPoint(p: torch.Tensor):
    """
    (B,V,3) -> (B,V)
    """
    return norm(p).squeeze(-1)

# Smooth boolean operations and sdf primitives by Inigo Quilez
# Modified for batched use

# The MIT License
# Copyright © 2018 Inigo Quilez
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

def sdUnion(d1: torch.Tensor, d2: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    (B,V), (B,V), (B,1) -> (B,V)
    """
    ka = torch.abs(k)
    h = torch.square(torch.clip(ka - torch.abs(d1 - d2), min=0))
    return torch.minimum(d1, d2) - h/(4*ka)


# https://iquilezles.org/articles/distfunctions/

def sdBox(p: torch.Tensor, h: torch.Tensor, w: torch.Tensor, l: torch.Tensor):
    """
    (B,V,3), (B,1,1), (B,1,1), (B,1,1) -> (B,V)
    """
    b = cat(h,w,l)
    q = torch.abs(p) - torch.abs(b)
    res = norm(torch.clip(q, min=0)) + torch.clip(torch.amax(q, dim=-1, keepdim=True), min=0)
    return res.squeeze(-1)

# modified from sdCappedTorus
def sdRingSegment(p: torch.Tensor, theta: torch.Tensor, r: torch.Tensor):
    """
    (B,V,3), (B,1,1), (B,1,1) -> (B,V)
    """
    px = torch.abs(p[:,:,0:1])
    py = p[:,:,1:2]
    
    half_theta = torch.clip(torch.abs(theta), max=torch.pi)/2
    c = torch.cos(half_theta)
    s = torch.sin(half_theta)
    k = torch.where(px*c > py*s, px*s + py*c, norm(p[:,:,:2]))
    q = norm2(p) + r*r - 2.0*torch.abs(r)*k
    res = torch.sqrt(q.clamp_min(0))
    return res.squeeze(-1)

# modified from sdVerticalCapsule
def sdSegment(p: torch.Tensor, h: torch.Tensor):
    """
    (B,V,3), (B,1,1) -> (B,V)
    """
    p2 = p.clone()
    p2[:,:,1:2] -= torch.clip(p[:,:,1:2], torch.zeros_like(h), torch.abs(h))
    return norm(p2).squeeze(-1)

def sdCappedCylinder(p: torch.Tensor, r: torch.Tensor, h: torch.Tensor):
    """
    (B,V,3), (B,1,1), (B,1,1) -> (B,V)
    """
    d = torch.abs(cat(norm(p[:,:,:2]), p[:,:,2:])) - torch.abs(cat(r, h))
    res = torch.clip(torch.amax(d, dim=-1, keepdim=True), max=0) + norm(torch.clip(d, min=0))
    return res.squeeze(-1)

def sdCappedCone(p: torch.Tensor, h: torch.Tensor, r1: torch.Tensor, r2: torch.Tensor):
    """
    (B,V,3), (B,1,1), (B,1,1), (B,1,1) -> (B,V)
    """
    ha = torch.abs(h)
    r1a = torch.abs(r1)
    r2a = torch.abs(r2)
    q = cat(norm(p[:,:,:2]), p[:,:,2:])
    k1 = cat(r2a, ha)
    k2 = cat(r2a-r1a, 2.0*ha)
    ca = cat(q[:,:,0:1] - torch.minimum(q[:,:,0:1], torch.where(q[:,:,1:2] < 0.0, r1a, r2a)), abs(q[:,:,1:2]) - ha)
    cb = q - k1 + k2*torch.clip(dot(k1-q, k2)/norm2(k2), 0, 1)
    s = torch.where((ca[:,:,0:1] < 0) & (cb[:,:,1:2] < 0), -1, 1)
    res = s*torch.sqrt(torch.minimum(norm2(ca), norm2(cb)))
    return res.squeeze(-1)