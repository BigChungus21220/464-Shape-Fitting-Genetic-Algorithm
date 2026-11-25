# main genetic algorithm loop

from reconstruction import reconstruct_sdf
import torch
import trimesh
from model import Population, SDF
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import shutil

shapes = [
    'datasets/shapes/dog.obj',
    'datasets/shapes/hand.obj',
    'datasets/shapes/pot.obj',
    'datasets/shapes/rod.obj',
    'datasets/shapes/sofa.obj'
]

mesh: trimesh.Trimesh = trimesh.load(shapes[3], force='mesh') # type: ignore

padding = 0.05 # percent padding to add to bounds
bounds: tuple[list[float], list[float]] = mesh.bounds
size = [bounds[1][i] - bounds[0][i] for i in range(3)]
lower = [(bounds[0][i] - size[i]*padding) for i in range(3)]
upper = [(bounds[1][i] + size[i]*padding) for i in range(3)]

#lower = (-1,-1,-1)
#upper = (2,2,2)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32
verts = torch.tensor(mesh.vertices, device=device, dtype=dtype).unsqueeze(0)

# 0.2219 sum square sdf on couch is current record

n_iter = 500
summary_count = 10
reconstruction_resolution = 200

fig_path = Path("output/loss_history.png")
models_path = Path("output/models")

models_path.mkdir(exist_ok=True, parents=True)

for item in models_path.iterdir():
    if item.is_dir():
        shutil.rmtree(item)
    else:
        item.unlink()


def simpson_index(N: int, n: torch.Tensor):
    return 1 - torch.sum(n*(n - 1)).cpu().item()/(N*(N - 1))


loss_history = np.zeros((n_iter, summary_count))
color_history = np.zeros((n_iter, summary_count, 3))

def callback(iter: int, max_iter: int, population: int, species_count: int, species_populations: torch.Tensor, losses: torch.Tensor, colors: torch.Tensor, sdfs: list[SDF]):
    """
    Args:
        iter (int): Generation number (starting at 1)
        max_iter (int): Maximum number of generations
        population (int): Overall population
        species_count (int): Number of species in total
        species_populations (torch.Tensor): Population of every species
        losses (torch.Tensor): Losses of top {summary_count} species representatives
        colors (torch.Tensor | None): Rgb colors of top {summary_count} species representatives
        sdfs (list[SDF]): SDF functions of top {summary_count} species representatives
    """
    
    loss_summary = ", ".join([f"{loss:.5g}" for loss in losses.cpu().tolist()])
    node_count = sum(len(sdf.nodes) for sdf in sdfs)/len(sdfs)
    print(f"Generation {iter}/{max_iter} -- Species: {species_count}, Avg Nodes: {node_count:.5g}, Simpson's Index: {simpson_index(population, species_populations):.5g}, Top {summary_count} species:\nLoss: {loss_summary}\n")
    
    
    loss_history[iter-1] = losses.cpu().numpy()
    if colors is not None:
        color_history[iter-1] = colors.cpu().numpy()

    if iter % 10 == 0:
        plot_loss(iter, max_iter)
        
    if iter % 50 == 0:
        write_model_files(iter, sdfs)


def plot_loss(iter: int, max_iter: int):
    y = loss_history[:iter]
    x = np.repeat(np.arange(iter)[:,None], summary_count, axis=(1))
    rgb = color_history[:iter]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        x.ravel(),
        y.ravel(),
        c=rgb.reshape(-1, 3),
        edgecolors="black",
        linewidths=0.6,
        s=36,
        alpha=0.9,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Population Loss History (Epoch {iter}/{max_iter})")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print("Loss plot updated.")


def write_model_files(iter: int, sdfs: list[SDF]):
    output_path = models_path/f"iter_{iter}"
    output_path.mkdir(parents=True, exist_ok=True)

    for i, sdf in enumerate(sdfs):
        output_json = output_path.joinpath(f"sdf-{i + 1}.json")
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(sdf.serialize(), f, indent=4)
            
        output_model = output_path.joinpath(f"sdf-{i + 1}.obj")
        verts, tris = reconstruct_sdf(sdf, lower, upper, reconstruction_resolution)
        mesh = trimesh.Trimesh(vertices=verts.cpu().tolist(), faces=tris.cpu().tolist())
        mesh.export(output_model)
        
    print("Models exported")
            
        


population = Population(
    pop_size = 1000,
    std = 1.0,
    p_add = 0.8,
    min_nodes = 5,
    max_nodes = 20,
    func_weights = [1.0]*6, # you can use this as a mask to disable specific functions
    device = device,
    dtype = dtype
)

population.train(
    verts,
    n_iter = n_iter,
    max_nodes_per_batch = 256,
    mut_p = 0.05,
    mut_std = 0.05,
    mut_max_angle = 0.01,
    callback = callback,
    summary_count = summary_count,
    do_fitness_sharing = True, # I had a mass extinction, leave this on
    keep_best = 2,
    species_thresh = 0.5
)