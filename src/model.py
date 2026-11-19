# crossover / mutation operators, gene definitions

from collections import defaultdict
from collections.abc import Callable
from typing import cast
from utils import quat_random, quat_apply
import torch
from sdf import SdfNode, sdUnion, sdSubtraction, sdBox, sdRingSegment, sdSegment, sdCappedCylinder, sdCappedCone, sdPoint
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
            
    # also add a chance to swap nodes

def crossover(a: list[SdfNode], b: list[SdfNode]):
    child = []
    for node in a:
        if random.random() < 0.5:
            child.append(node.clone())
            
    for node in b:
        if random.random() < 0.5:
            child.append(node.clone())
            
    return child

# functions taking a (B, V, 3) tensor of sample points, and n_params (B, 1, 1) tensors of input params returning a (B, V) tensor
# function, n_params pairs
sdf_functions = {
    sdBox: 3,
    sdRingSegment: 2,
    sdSegment: 1,
    sdCappedCylinder: 2,
    sdCappedCone: 3,
    sdPoint: 0
}

class Population:
    # P: population size
    # N: population node count
    # C: number of color dimensions
    # V: number of model vertices
    
    device: torch.device
    dtype: torch.dtype
    population: list[list[int]] # [P,] tensor, node indices
    node_counts: torch.Tensor # (P,) int, node count of each individual
    colors: torch.Tensor # (P, C) float, individual "colors", tracks speciation

    ids: torch.Tensor # (N,) float, node ids, tracks uniqueness
    rotations: torch.Tensor # (N, 1, 4) float, quaternion rotation applied to input points
    translations: torch.Tensor # (N, 1, 3) float, offset applied to input points
    inflates: torch.Tensor # (N, 1) float, offset applied to the zero of the sdf
    smoothings: torch.Tensor # (N, 1) float, smoothing factors to apply
    signs: torch.Tensor # (N, 1) float {-1, 1}, whether each node is additive or subtractive
    
    # map node index to function type and param index
    param_map: list[tuple[Callable[..., torch.Tensor], int]]
    # map from sdf function type to parameters for all nodes with that type
    params: dict[Callable[..., torch.Tensor], torch.Tensor]
        
    
    def __init__(self, pop_size: int, device: torch.device | str, dtype: torch.dtype):
        std = 1.0
        p_add = 0.8
        min_nodes = 5
        max_nodes = 20
        func_weights = [1.0]*6
        
        self.dtype = dtype
        self.device = torch.device(device)
        
        # figure out how many nodes each individual has
        node_count_torch = torch.randint(min_nodes, max_nodes, (pop_size,), device='cpu')
        node_count = cast(list[int], node_count_torch.tolist())
        
        nodes_end = cast(list[int], torch.cumsum(node_count_torch, dim=0)[:-1].tolist())
        
        # count total nodes in population
        N = cast(int, torch.sum(node_count_torch).cpu().item())
        
        # generate transforms
        self.ids = torch.rand(size=(N,), device=device, dtype=dtype)
        self.rotations = quat_random(N, torch.pi, device=device, dtype=dtype)
        self.translations = torch.normal(mean=0, std=std, size=(N, 1, 3), device=device, dtype=dtype)
        self.inflates = torch.normal(mean=0, std=std, size=(N, 1), device=device, dtype=dtype)
        self.smoothings = torch.normal(mean=0, std=std, size=(N, 1), device=device, dtype=dtype)
        self.signs = torch.where(torch.rand((N, 1)) < p_add, 1.0, -1.0)
        
        # generate population
        func_frequencies = {func: 0 for func, n_params in sdf_functions.items()}
        self.population = []
        self.node_counts = torch.tensor((pop_size,), device=device, dtype=torch.int32)
        
        # me when I have to use a loop in python :(
        for i in range(pop_size):
            # choose node types
            funcs = random.choices(list(sdf_functions.keys()), weights=func_weights, k=node_count[i])
            # get the param list indices
            for func in funcs:
                self.param_map.append((func, func_frequencies[func]))
                func_frequencies[func] += 1
                
            self.node_counts[i] = node_count[i]
            self.population.append(list(range(nodes_end[i] - node_count[i], nodes_end[i])))
        
        # generate function params
        self.params = {
            func: torch.normal(mean=0, std=std, size=(func_frequencies[func], n_params, 1, 1), device=device, dtype=dtype) 
            for func, n_params in sdf_functions.items()
        }

    def evaluate(self, verts: torch.Tensor):
        """
        (1, V, 3) -> (N, 1, 1)
        """
        transformed = quat_apply(self.rotations, verts) - self.translations # (N, V, 3)
        
        batches = self.gen_batches(16)
        fitnesses = torch.tensor((len(self.population),), device=self.device, dtype=self.dtype)
        
        for batch in batches:
            points = transformed[batch]
            
            batch_pop = [self.population[i] for i in batch]
            
            # maps from func types to indices
            func_nodes = defaultdict(list) # list of (index of node in batch, index of node in population, index of node in population func group)
            batch_idx = 0
            for indiv in batch_pop:
                for node_idx in indiv:
                    node_type, param_idx = self.param_map[node_idx]
                    func_nodes[node_type].append((batch_idx, node_idx, param_idx))
                    batch_idx += 1
            
            node_count = sum([len(nodes) for nodes in func_nodes.values()])
            
            # eval each type
            node_sdfs = torch.tensor((node_count, verts.shape[1]), device=self.device, dtype=self.dtype) # (L, V)
            for func, nodes in func_nodes.items():
                batch_idx, node_idx, param_idx = torch.tensor(nodes, device=self.device, dtype=torch.int32).unbind(1)
                params = self.params[func][param_idx] # (H, P, 1, 1)
                samples = points[node_idx] # (H, V, 3)
                node_sdfs[batch_idx] = func(samples, *params)
                
            # get individual sdfs
            indiv_sdfs = torch.tensor((len(batch_pop), verts.shape[1]), device=self.device, dtype=self.dtype) # (B, V)
            batch_idx = 0
            for i in range(len(batch_pop)):
                sdf = torch.full((1, verts.shape[1]), torch.inf, device=self.device, dtype=self.dtype)
                for node_idx in self.population[i]:
                    indiv_sdfs[node_idx] = self.signs[node_idx]*sdUnion(sdf, self.signs[node_idx]*node_sdfs[batch_idx] - self.inflates[node_idx], self.smoothings[node_idx])
                    batch_idx += 1
                    
            # take fitnesses
            fitnesses[batch] = torch.sum(torch.square(indiv_sdfs), dim=1)
            
        return fitnesses
        
    def gen_batches(self, k: int):
        # assume individuals with a similar color have similar nodes
        # try to group those first to maximize number of nodes evaluated in parallel
        
        batches: list[torch.Tensor] = []
        P, D = self.colors.shape
        mask = torch.zeros(P, device=self.device, dtype=torch.bool)
        while torch.any(mask):
            # find a new medoid
            medoid = self.colors[torch.argmax(mask), :]
            rem = self.colors[mask]
            # evaluate similarity
            dist = torch.sum(rem*medoid, dim=-1)/torch.linalg.vector_norm(rem, dim=-1)
                        
            # add to group until full
            batch = []
            rem_nodes = k
            size_mask = self.node_counts < rem_nodes
            while torch.any(size_mask):
                _, top_index = torch.max(dist[size_mask], dim=0)
                batch.append(top_index)
                rem_nodes -= self.node_counts[top_index]
                size_mask = self.node_counts < rem_nodes
                
            batch_indices = torch.tensor(batch, device=self.device, dtype=torch.int32)
            
            mask[batch_indices] = True
            batches.append(batch_indices)  
                  
        return batches