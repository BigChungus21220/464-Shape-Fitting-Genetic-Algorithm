# sdf definitions and operators

from typing import Protocol
from collections.abc import Callable
from dataclasses import dataclass
import torch
import random

# B: batch size (probably always 1 unless we do some real fancy optimizations)
# V: num gt vertices

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

def transform(p: torch.Tensor, t: torch.Tensor, R: torch.Tensor):
    """
    Applies R^-1*(p - t)
    (1,V,3), (B,1,3), (B,1,3,3) -> (B,V,3)
    """
    p -= t
    return (R.transpose(-1, -2) @ p.unsqueeze(-1)).squeeze(-1)

def random_rotation(batch_size: int, max_angle: float, device: torch.device | str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Generates random 3D rotation matrices (uniform in angle applied)
     -> (B,1,3,3)
    """
    axis = torch.randn((batch_size, 3), device=device, dtype=dtype)
    axis = torch.nn.functional.normalize(axis, p=2, dim=-1)

    half_theta = 0.5 * max_angle * torch.rand((batch_size,), device=device, dtype=dtype)

    w = torch.cos(half_theta)
    x, y, z = (axis * torch.sin(half_theta).unsqueeze(-1)).unbind(-1)
    
    # quaternions to rotation matrices
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    
    R = torch.stack([
        1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy),
        2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),
        2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)
    ], dim=-1)
    
    return R.view(batch_size, 3, 3).unsqueeze(1)

# Smooth boolean operations and sdf primitives by Inigo Quilez

# The MIT License
# Copyright © 2018 Inigo Quilez
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

def sdUnion(d1: torch.Tensor, d2: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    (B,V), (B,V), (B,1) -> (B,V)
    """
    h = torch.square(torch.clip(k - torch.abs(d1 - d2), min=0))
    return torch.minimum(d1, d2) - h/(4*k)

def sdSubtraction(d1: torch.Tensor, d2: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    (B,V), (B,V), (B,1) -> (B,V)
    """
    return -sdUnion(d1,-d2,k)

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
    p[:,:,0] = torch.abs(p[:,:,0])
    
    half_theta = torch.clip(torch.abs(theta), max=torch.pi)/2
    xc = torch.cos(half_theta)*p[:,:,0]
    ys = torch.sin(half_theta)*p[:,:,1]
    k = torch.where(xc > ys, xc + ys, norm(p[:,:,:2]))
    res = torch.sqrt(norm2(p) + r*r - 2.0*torch.abs(r)*k)
    return res.squeeze(-1)

# modified from sdVerticalCapsule
def sdSegment(p: torch.Tensor, h: torch.Tensor):
    """
    (B,V,3), (B,1,1) -> (B,V)
    """
    p[:,:,1] -= torch.clip(p[:,:,1], torch.zeros_like(h), torch.abs(h))
    return norm(p).squeeze(-1)

def sdCappedCylinder(p: torch.Tensor, r: torch.Tensor, h: torch.Tensor):
    """
    (B,V,3), (B,1,1), (B,1,1) -> (B,V)
    """
    d = torch.abs(cat(norm(p[:,:,:2]), p[:,:,2])) - torch.abs(cat(r, h))
    res = torch.clip(torch.amax(d, dim=-1, keepdim=True), max=0) + norm(torch.clip(d, min=0))
    return res.squeeze(-1)

def sdCappedCone(p: torch.Tensor, h: torch.Tensor, r1: torch.Tensor, r2: torch.Tensor):
    """
    (B,V,3), (B,1,1), (B,1,1), (B,1,1) -> (B,V)
    """
    h = torch.abs(h)
    r1 = torch.abs(r1)
    r2 = torch.abs(r2)
    q = cat(norm(p[:,:,:2]), p[:,:,2])
    k1 = cat(r2, h)
    k2 = cat(r2-r1, 2.0*h)
    ca = cat(q[:,:,0] - torch.minimum(q[:,:,0], torch.where(q[:,:,1] < 0.0, r1, r2)), abs(q[:,:,1]) - h)
    cb = q - k1 + k2*torch.clip(dot(k1-q, k2)/norm2(k2), 0, 1)
    s = torch.where((cb[:,:,0] < 0) & (cb[:,:,1] < 0), -1, 1)
    res = s*torch.sqrt(torch.minimum(norm2(ca), norm2(cb)))
    return res.squeeze(-1)

# obv this one doesn't belong to anyone
def sdPoint(p: torch.Tensor):
    """
    (B,V,3) -> (B,V)
    """
    return norm(p).squeeze(-1)



    
sdf_functions = [
    (sdBox, 3),
    (sdRingSegment, 2),
    (sdSegment,1),
    (sdCappedCylinder, 2),
    (sdCappedCone, 3),
    (sdPoint, 0)
]

class SdfNode:
    dtype: torch.dtype
    rotation: torch.Tensor # (1,1,3,3)
    translation: torch.Tensor # (1,1,3)
    inflate: torch.Tensor # (1,1,1)
    additive: bool
    func: Callable[..., torch.Tensor]
    params: torch.Tensor # tensors (n_params,1,1,1)
    
    def random(self, std: float, p_add: float, func_weights: list[float], device: torch.device | str, dtype: torch.dtype):
        """
            func (SdfCallback): Sdf function to use
            n_params (int): Number of additional parameters to the sdf
            std (float): Standard deviation to use when generating initial values
            p_add (float): Probability shape will be additive
            func_weights (list[float]): Weights to apply when choosing underlying sdf function 
        """
        func, n_params = random.choices(sdf_functions, weights=func_weights)[0]
        self.func = func
        
        self.additive = random.random() < p_add
        self.rotation = random_rotation(1, torch.pi, device, dtype)
        self.translation = torch.normal(mean=0, std=std, size=(1, 1, 3), device=device, dtype=dtype)
        self.inflate = torch.normal(mean=0, std=std, size=(1, 1, 1), device=device, dtype=dtype)
        self.params = torch.normal(mean=0, std=std, size=(n_params, 1, 1, 1), device=device, dtype=dtype)
        
        self.device = self.rotation.device
        self.dtype = dtype
    
    def evaluate(self, p: torch.Tensor):
        """
        Evaluates the sdf at each point
        (1,V) -> (1,V)
        """
        p = transform(p, self.translation, self.rotation)
        return self.func(p, *self.params.unbind(0)) - torch.abs(self.inflate)
    
    def mutate(self, std: float, max_angle: float):
        """
            std (float): Standard deviation to apply to float parameters
            max_angle (float): Max change in rotation
        """
        self.rotation = random_rotation(1, max_angle, device=self.device, dtype=self.dtype) @ self.rotation
        
        self.translation += torch.normal(mean=0, std=std, size=(1, 1, 3), device=self.device, dtype=self.dtype)
        self.inflate += torch.normal(mean=0, std=std, size=(1, 1, 1), device=self.device, dtype=self.dtype)
        self.params += torch.normal(mean=0, std=std, size=self.params.shape, device=self.device, dtype=self.dtype)