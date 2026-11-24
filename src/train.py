# main genetic algorithm loop

import torch
import trimesh
from model import Population, SDF
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

shapes = [
    'datasets/shapes/dog.obj',
    'datasets/shapes/hand.obj',
    'datasets/shapes/pot.obj',
    'datasets/shapes/rod.obj',
    'datasets/shapes/sofa.obj'
]

mesh: trimesh.Trimesh = trimesh.load(shapes[0], force='mesh') # type: ignore

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32
verts = torch.tensor(mesh.vertices, device=device, dtype=dtype).unsqueeze(0)

# 0.2219 sum square sdf on couch is current record

n_iter = 500
summary_count = 10

loss_history = np.zeros((n_iter, summary_count))
color_history = np.zeros((n_iter, summary_count, 3))
fig_path = Path("output/loss_history.png")

def callback(iter: int, max_iter: int, losses: torch.Tensor, colors: torch.Tensor | None, sdfs: list[SDF]):
    summary = ", ".join([f"{loss:.5g}" for loss in losses.cpu().tolist()])
    node_count = sum(len(sdf.nodes) for sdf in sdfs)/len(sdfs)
    print(f"Epoch {iter}/{max_iter} -- avg nodes: {node_count}, loss: [{summary}]")
    loss_history[iter-1] = losses.cpu().numpy()
    if colors is not None:
        color_history[iter-1] = colors.cpu().numpy()

    if iter % 10 == 0:
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

population.train(
    verts,
    n_iter = n_iter,
    max_nodes_per_batch = 256,
    mut_p = 0.01,
    mut_std = 0.01,
    mut_max_angle = 0.005,
    callback = callback,
    summary_count = summary_count
)