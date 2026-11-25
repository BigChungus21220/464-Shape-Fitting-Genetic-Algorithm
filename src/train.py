# main genetic algorithm loop

import torch
import trimesh
import json
from model import Population, SDF
import reconstruction
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timezone

shapes_directory = 'datasets/shapes/'

shapes = [
    'dog.obj',
    'hand.obj',
    'pot.obj',
    'rod.obj',
    'sofa.obj'
]

# 0.2219 sum square sdf on couch is current record

timestamp_initial = datetime.fromtimestamp(datetime.now().timestamp())

n_iter = 500
summary_count = 10

loss_history = np.zeros((n_iter, summary_count))
color_history = np.zeros((n_iter, summary_count, 3))
fig_path = Path("output/loss_history.png")
models_path = Path("output")



def callback(iter: int, max_iter: int, losses: torch.Tensor, colors: torch.Tensor | None, sdfs: list[SDF], epoch_size: int = 10):
    print(f"Epoch {iter}/{max_iter}: [{", ".join([f"{loss:.5g}" for loss in losses.cpu().tolist()])}]")
    loss_history[iter-1] = losses.cpu().numpy()
    if colors is not None:
        color_history[iter-1] = colors.cpu().numpy()

    if iter % epoch_size == 0:
        y = loss_history[:iter]
        x = np.repeat(np.arange(iter)[:,None], summary_count, axis=(1))
        rgb = color_history[:iter]

        print(rgb.size)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(
            x.ravel(),
            y.ravel(),
            c=rgb.reshape(-1, 3).clip(0, 1),
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

        write_model_files(iter=iter, sdfs=sdfs)
        
        print("Loss plot updated.")

def write_model_files(iter: int, sdfs: list[SDF]):
    output_path = models_path/f"iter_{iter}"
    output_path.mkdir(parents=True, exist_ok=True)

    for sdf in sdfs:
        output_file_path = output_path.joinpath(f"sdf-{sdf.idx}.json")
        with output_file_path.open("w", encoding="utf-8") as f:
            json.dump(sdf.serialize(), f, indent=4)    


if __name__ == "__main__":

    print("starting training")

    timestamp_initial = datetime.fromtimestamp(datetime.now().timestamp())
    
    for shape in shapes:
        '''print(f"Training shape: {shape}")
        shape_path = shapes_directory + shape

        mesh: trimesh.Trimesh = trimesh.load(shape_path, force='mesh') # type: ignore

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float32
        verts = torch.tensor(mesh.vertices, device=device, dtype=dtype).unsqueeze(0)

        loss_history = np.zeros((n_iter, summary_count))
        color_history = np.zeros((n_iter, summary_count, 3))
        fig_path = Path("output/loss_history.png")
        models_path = Path("output")

                
        fig_path = Path(f"output/{timestamp_initial}/loss_history_{shape}.png")
        models_path = Path(f"output/{timestamp_initial}/sdfs/{shape}")
        models_path.mkdir(parents=True, exist_ok=True)
    
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
    
        print("Training Completed, beginning example reconstruction")'''
        models_path = Path(f"output/2026-11-25 01:23:50.863517/sdfs/{shape}")
    
        subdirs = (entry for entry in models_path.iterdir() if entry.is_dir())
    
        for index, subdir in reversed(list(enumerate(subdirs))):
            print(f"{index}: {subdir}")
            for filepath in subdir.iterdir():
                if filepath.is_file() and str(filepath).endswith(".json"):
                    print(filepath)
                    reconstruction.recreate_obj_file(str(filepath))