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

def simpson_index(N: int, n: torch.Tensor):
    return 1 - torch.sum(n*(n - 1)).cpu().item()/(N*(N - 1))


# 0.2219 sum square sdf on couch is current record

def run(
    shape: str = shapes[0],
    n_iter: int = 500,
    pop_size: int = 1000,
    summary_count: int = 10,
    min_nodes: int = 5,
    max_nodes: int = 20,
    mut_p: float = 0.01,
    mut_std: float = 0.01,
    mut_max_angle: float = 0.005,
    keep_best: int = 2,
    species_thresh: float = 0.5,
    fitness_sharing: bool = True, # I had a mass extinction, leave this on
    max_nodes_per_batch: int = 256,
    plot_every: int = 10,
    fig_path: str | Path = "output/loss_history.png",
    device: str | None = None,
    seed: int | None = None,
):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    mesh: trimesh.Trimesh = trimesh.load(shape, force='mesh') # type: ignore
    verts = torch.tensor(mesh.vertices, device=device, dtype=dtype).unsqueeze(0)

    loss_history = np.zeros((n_iter, summary_count))
    color_history = np.zeros((n_iter, summary_count, 3))

    def callback(iter: int, max_iter: int, population: int, species_count: int, species_populations: torch.Tensor, losses: torch.Tensor, colors: torch.Tensor | None, sdfs: list[SDF]):
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

        if iter % plot_every == 0 or iter == max_iter:
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
        pop_size = pop_size,
        std = 1.0,
        p_add = 0.8,
        min_nodes = min_nodes,
        max_nodes = max_nodes,
        func_weights = [1.0]*6,
        device = device,
        dtype = dtype
    )

    population.train(
        verts,
        n_iter = n_iter,
        max_nodes_per_batch = max_nodes_per_batch,
        mut_p = mut_p,
        mut_std = mut_std,
        mut_max_angle = mut_max_angle,
        callback = callback,
        summary_count = summary_count,
        do_fitness_sharing = fitness_sharing,
        keep_best = keep_best,
        species_thresh = species_thresh
    )
    return population

# Alternative run method
# python src/train.py
if __name__ == "__main__":
    run()
