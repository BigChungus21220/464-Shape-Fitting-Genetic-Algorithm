# crossover / mutation operators, gene definitions

import torch
from sdf import SdfNode, sdUnion, sdSubtraction
import random

def generate(min_nodes: int, max_nodes: int, device: torch.device | str, dtype: torch.dtype):
    node_count = random.randint(min_nodes, max_nodes)
    return [SdfNode.random(1.0, 0.8, [1.0]*6, device, dtype) for i in range(node_count)]

def evaluate(x: list[SdfNode], verts: torch.Tensor):
    sdf = torch.full((1, verts.shape[1]), torch.inf, device=verts.device, dtype=verts.dtype)
    for node in x:
        if node.additive:
            sdf = sdUnion(sdf, node.evaluate(verts), node.smoothing)
        else:
            sdf = sdSubtraction(sdf, node.evaluate(verts), node.smoothing)
        
    return torch.sum(torch.square(sdf), dim=1)

def mutate(x: list[SdfNode], p: float):
    """
    p is the probability of mutation.
    since mutation occurs per node, there is an implicit bias towards shorter solutions
    """
    for node in x:
        if random.random() < p:
            node.mutate(0.01, 0.005)

def crossover(a: list[SdfNode], b: list[SdfNode]):
    child = []
    for node in a:
        if random.random() < 0.5:
            child.append(node.clone())
            
    for node in b:
        if random.random() < 0.5:
            child.append(node.clone())
            
    return child
        