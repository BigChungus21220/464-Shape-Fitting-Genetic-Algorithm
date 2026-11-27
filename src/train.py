# main genetic algorithm loop

from mesh_utils import reconstruct_sdf, resample_mesh
from metrics import ShapeMetrics
import torch
import trimesh
from model import Population, SDF
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

shapes = [
    'datasets/shapes/dog.obj',
    'datasets/shapes/hand.obj',
    'datasets/shapes/pot.obj',
    'datasets/shapes/rod.obj',
    'datasets/shapes/sofa.obj'
]

def simpson_index(N: int, n: torch.Tensor):
    return 1 - torch.sum(n*(n - 1)).detach().cpu().item()/(N*(N - 1))

def run(
    shape: str = shapes[3],
    resample: int | None = 1000,
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
    fitness_sharing: bool = False,
    max_nodes_per_batch: int = 256,
    refine_every: int = 0, # 0 = no gradient refinement (baseline run)
    refine_steps: int = 5,
    refine_lr: float = 1e-2,
    final_top_k: int = 5,
    final_steps: int = 300,
    plot_every: int = 10,
    save_every: int = 50,
    out_dir: str | Path = "output",
    padding: float = 0.05, # percent padding to add to reconstruction bounds
    reconstruction_resolution: int = 200,
    metrics_resolution: int = 64, # grid resolution for the chamfer reconstruction each generation
    metrics_surface_samples: int = 5000, # surface points per shape for chamfer distance
    metrics_volume_samples: int = 50000, # points in the bounding box for IoU
    device: str | None = None,
    seed: int | None = None,
):
    config = {k: str(v) for k, v in locals().items()}

    # each run gets its own timestamped folder: <out_dir>/<YYYY-MM-DD_HH-MM-SS>/
    run_dir = Path(out_dir)/datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    fig_path = run_dir/"loss_history.png"
    models_path = run_dir/"models"
    models_path.mkdir()
    with (run_dir/"config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"Writing output to {run_dir}")

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    mesh: trimesh.Trimesh = trimesh.load(shape, force='mesh') # type: ignore
    bounds: tuple[list[float], list[float]] = mesh.bounds
    size = [bounds[1][i] - bounds[0][i] for i in range(3)]
    lower = [(bounds[0][i] - size[i]*padding) for i in range(3)]
    upper = [(bounds[1][i] + size[i]*padding) for i in range(3)]
    
    if resample is None:
        verts = torch.tensor(mesh.vertices, device=device, dtype=dtype).unsqueeze(0)
        normals = torch.tensor(mesh.vertex_normals, device=device, dtype=dtype).unsqueeze(0)
    else:
        tris = torch.tensor(mesh.triangles, device=device, dtype=dtype)
        tri_normals = torch.tensor(mesh.face_normals, device=device, dtype=dtype)
        verts, normals = resample_mesh(tris, tri_normals, resample)
        verts = verts.unsqueeze(0)
        normals = normals.unsqueeze(0)

    print("Precomputing target samples for chamfer / IoU metrics...")
    metrics = ShapeMetrics(mesh, lower, upper, metrics_surface_samples, metrics_volume_samples, metrics_resolution, device, dtype)
    metrics_path = run_dir/"metrics.csv"
    metrics_fig_path = run_dir/"metrics.png"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write("generation,loss,chamfer,iou\n")

    loss_history = np.zeros((n_iter, summary_count))
    chamfer_history = np.full(n_iter, np.nan)
    iou_history = np.full(n_iter, np.nan)
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
        
        # chamfer distance and IoU of the best individual against the target
        m = metrics(sdfs[0])
        chamfer_history[iter-1] = m["chamfer"]
        iou_history[iter-1] = m["iou"]
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(f"{iter},{losses[0].item():.8g},{m['chamfer']:.8g},{m['iou']:.8g}\n")

        loss_summary = ", ".join([f"{loss:.5g}" for loss in losses.detach().cpu().tolist()])
        node_count = sum(len(sdf.nodes) for sdf in sdfs)/len(sdfs)
        print(f"Generation {iter}/{max_iter} -- Species: {species_count}, Avg Nodes: {node_count:.5g}, Simpson's Index: {simpson_index(population, species_populations):.5g}, Top {summary_count} species:\nLoss: {loss_summary}\nBest -- Chamfer: {m['chamfer']:.5g}, IoU: {m['iou']:.5g}\n")
        loss_history[iter-1] = losses.detach().cpu().numpy()
        if colors is not None:
            color_history[iter-1] = colors.detach().cpu().numpy()

        if iter % plot_every == 0 or iter == max_iter:
            plot_loss(iter, max_iter)
            plot_metrics(iter, max_iter)
            
        if iter % save_every == 0:
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

    
    def plot_metrics(iter: int, max_iter: int):
        x = np.arange(1, iter + 1)
        fig, (ax_cd, ax_iou) = plt.subplots(1, 2, figsize=(14, 5))
        ax_cd.plot(x, chamfer_history[:iter])
        ax_cd.set_yscale("log")
        ax_cd.set_xlabel("Epoch")
        ax_cd.set_ylabel("Chamfer distance")
        ax_cd.set_title("Best individual: Chamfer distance")
        ax_cd.grid(True, which="both", linestyle="--", alpha=0.3)
        ax_iou.plot(x, iou_history[:iter])
        ax_iou.set_ylim(0, 1)
        ax_iou.set_xlabel("Epoch")
        ax_iou.set_ylabel("IoU")
        ax_iou.set_title("Best individual: IoU")
        ax_iou.grid(True, linestyle="--", alpha=0.3)
        fig.suptitle(f"Metrics (Epoch {iter}/{max_iter})")
        fig.savefig(metrics_fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def write_model_files(iter: int | str, sdfs: list[SDF]):
        output_path = models_path/f"iter_{iter}"
        output_path.mkdir(parents=True, exist_ok=True)

        for i, sdf in enumerate(sdfs):
            output_json = output_path.joinpath(f"sdf-{i + 1}.json")
            with output_json.open("w", encoding="utf-8") as f:
                json.dump(sdf.serialize(), f, indent=4)
                
            output_model = output_path.joinpath(f"sdf-{i + 1}.obj")
            verts, tris = reconstruct_sdf(sdf, lower, upper, reconstruction_resolution)
            mesh = trimesh.Trimesh(vertices=verts.detach().cpu().tolist(), faces=tris.detach().cpu().tolist())
            mesh.export(output_model)
            
        print("Models exported")
            
        


    extent = verts.abs().amax().item() # largest half-width of the target
    population = Population(
        pop_size = pop_size,
        std = 0.25*extent,
        p_add = 0.9,
        min_nodes = min_nodes,
        max_nodes = max_nodes,
        func_weights = [1.0]*6, # you can use this as a mask to disable specific functions
        device = device,
        dtype = dtype
    )

    population.train(
        verts,
        normals,
        n_iter = n_iter,
        max_nodes_per_batch = max_nodes_per_batch,
        mut_p = mut_p,
        mut_std = mut_std,
        mut_max_angle = mut_max_angle,
        callback = callback,
        summary_count = summary_count,
        do_fitness_sharing = fitness_sharing,
        keep_best = keep_best,
        species_thresh = species_thresh,
        refine_every = refine_every,
        refine_steps = refine_steps,
        refine_lr = refine_lr,
    )

    if final_steps > 0 and final_top_k > 0:
        population.refine_top(
            verts,
            normals,
            max_nodes_per_batch = max_nodes_per_batch,
            top_k = final_top_k,
            steps = final_steps,
            lr = refine_lr,
        )
        refined_sdfs = [
        population.get_individual(i)
        for i in range(min(final_top_k, len(population.population)))
        ]
        write_model_files("final_refined", refined_sdfs)
        final_metrics = [metrics(sdf) for sdf in refined_sdfs]
        for i, m in enumerate(final_metrics):
            print(f"Refined sdf-{i + 1} -- Chamfer: {m['chamfer']:.5g}, IoU: {m['iou']:.5g}")
        with (run_dir/"final_refined_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(final_metrics, f, indent=4)

    return population

# Alternative run method
# python src/train.py
if __name__ == "__main__":
    run()
