# main genetic algorithm loop

import torch
import trimesh
from model import Population

mesh: trimesh.Trimesh = trimesh.load('datasets/shapes/sofa.obj', force='mesh') # type: ignore

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32
verts = torch.tensor(mesh.vertices, device=device, dtype=dtype).unsqueeze(0)

# 0.2219 sum square sdf on couch is current record

population = Population(
    pop_size = 1000,
    std = 1.0,
    p_add = 0.8,
    min_nodes = 5,
    max_nodes = 20,
    func_weights = [1.0]*6,
    device = device,
    dtype = dtype
)

def callback(iter: int, max_iter: int, top_losses: torch.Tensor):
    print(f"Epoch {iter}/{max_iter}: [{", ".join([f"{loss:.5g}" for loss in top_losses.cpu().tolist()])}]")

population.train(
    verts,
    n_iter = 10000,
    max_nodes_per_batch = 256,
    mut_p = 0.01,
    mut_std = 0.01,
    mut_max_angle = 0.005,
    callback = callback,
    summary_count = 10
)