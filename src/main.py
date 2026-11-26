# entry point: python src/main.py --help

import argparse
from pathlib import Path

from train import run, shapes

SHAPE_NAMES = {Path(s).stem: s for s in shapes}


def main():
    parser = argparse.ArgumentParser(description="Fit a shape with a genetic algorithm.")
    io = parser.add_argument_group("input / output")
    io.add_argument("--shape", default="dog",
                    help=f"one of {list(SHAPE_NAMES)} or a path to a mesh file (default: dog)")
    io.add_argument("--out", default="output/loss_history.png", help="loss plot output path")
    io.add_argument("--plot-every", type=int, default=10, help="update the loss plot every N generations")
    io.add_argument("--summary-count", type=int, default=10, help="number of top species to report")

    ga = parser.add_argument_group("genetic algorithm")
    ga.add_argument("-n", "--generations", type=int, default=500, help="number of generations/epochs")
    ga.add_argument("-p", "--pop-size", type=int, default=1000, help="population size")
    ga.add_argument("--min-nodes", type=int, default=5, help="min primitives per initial individual")
    ga.add_argument("--max-nodes", type=int, default=20, help="max primitives per initial individual")
    ga.add_argument("--keep-best", type=int, default=2, help="elites carried over each generation")

    mut = parser.add_argument_group("mutation")
    mut.add_argument("--mut-p", type=float, default=0.01, help="per-node mutation probability")
    mut.add_argument("--mut-std", type=float, default=0.01, help="mutation standard deviation")
    mut.add_argument("--mut-max-angle", type=float, default=0.005, help="max rotation mutation (radians)")

    sp = parser.add_argument_group("speciation")
    sp.add_argument("--species-thresh", type=float, default=0.5, help="color distance threshold for species")
    sp.add_argument("--no-fitness-sharing", action="store_true", help="disable fitness sharing")

    rt = parser.add_argument_group("runtime")
    rt.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    rt.add_argument("--seed", type=int, default=None, help="random seed")
    args = parser.parse_args()
    shape = SHAPE_NAMES.get(args.shape, args.shape)
    assert shape is not None
    run(
        shape=shape,
        n_iter=args.generations,
        pop_size=args.pop_size,
        summary_count=args.summary_count,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        mut_p=args.mut_p,
        mut_std=args.mut_std,
        mut_max_angle=args.mut_max_angle,
        keep_best=args.keep_best,
        species_thresh=args.species_thresh,
        fitness_sharing=not args.no_fitness_sharing,
        plot_every=args.plot_every,
        fig_path=args.out,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
