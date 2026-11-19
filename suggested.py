from typing import Callable
import torch

SdfFunc = Callable[..., torch.Tensor]

# Ordered list of (function, num_params); index in this list is the integer func_id
SDF_SPECS: list[tuple[SdfFunc, int]] = [
    (sdBox, 3),
    (sdRingSegment, 2),
    (sdSegment, 1),
    (sdCappedCylinder, 2),
    (sdCappedCone, 3),
    (sdPoint, 0),
]
MAX_PARAMS = max(n for _, n in SDF_SPECS)


class Population:
    def __init__(
        self,
        pop_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        color_dim: int = 8,
        min_nodes: int = 5,
        max_nodes: int = 20,
        std: float = 1.0,
        p_add: float = 0.8,
        func_weights: list[float] | None = None,
    ):
        self.device = torch.device(device)
        self.dtype = dtype

        # 1. Node counts per individual & population slices
        self.node_counts = torch.randint(
            min_nodes, max_nodes + 1, (pop_size,), device=self.device, dtype=torch.int64
        )
        counts_cpu = self.node_counts.tolist()
        N = sum(counts_cpu)
        self.population: list[torch.Tensor] = list(
            torch.split(torch.arange(N, device=self.device), counts_cpu)
        )

        # 2. Speciation colors (P, C)
        self.colors = torch.randn((pop_size, color_dim), device=self.device, dtype=dtype)

        # 3. Vectorized per-node attributes (all indexed identically by 0..N-1)
        weights = torch.tensor(
            func_weights or [1.0] * len(SDF_SPECS), device=self.device, dtype=dtype
        )
        self.func_ids = torch.multinomial(weights, N, replacement=True)  # (N,)
        self.ids = torch.rand((N,), device=self.device, dtype=dtype)
        self.rotations = quat_random(N, torch.pi, device=self.device, dtype=dtype)
        self.translations = torch.normal(0.0, std, (N, 1, 3), device=self.device, dtype=dtype)
        self.inflates = torch.normal(0.0, std, (N, 1), device=self.device, dtype=dtype)
        self.smoothings = torch.normal(0.0, std, (N, 1), device=self.device, dtype=dtype)
        self.signs = torch.where(
            torch.rand((N, 1), device=self.device) < p_add, 1.0, -1.0
        ).to(dtype)
        self.params = torch.normal(
            0.0, std, (N, MAX_PARAMS, 1, 1), device=self.device, dtype=dtype
        )

    def evaluate(self, verts: torch.Tensor, max_nodes_per_batch: int = 256) -> torch.Tensor:
        """
        verts: (1, V, 3) -> returns (P,) MSE fitness per individual
        """
        V = verts.shape[1]
        transformed = quat_apply(self.rotations, verts) - self.translations  # (N, V, 3)
        fitnesses = torch.empty((len(self.population),), device=self.device, dtype=self.dtype)

        for batch_indivs in self.gen_batches(max_nodes_per_batch):
            # Gather global node indices for all individuals in this batch
            batch_node_lists = [self.population[i] for i in batch_indivs.tolist()]
            batch_nodes = torch.cat(batch_node_lists)  # (L,)
            L = batch_nodes.numel()

            # Evaluate all nodes in the batch grouped by SDF primitive type
            node_sdfs = torch.empty((L, V), device=self.device, dtype=self.dtype)
            batch_func_ids = self.func_ids[batch_nodes]

            for f_id, (func, n_params) in enumerate(SDF_SPECS):
                local_mask = batch_func_ids == f_id
                if not local_mask.any():
                    continue
                global_idx = batch_nodes[local_mask]
                samples = transformed[global_idx]  # (H, V, 3)
                params = self.params[global_idx, :n_params].unbind(1)  # n_params x (H, 1, 1)
                node_sdfs[local_mask] = func(samples, *params)

            # Apply per-node signed offset once before reduction
            signed_sdfs = self.signs[batch_nodes] * (node_sdfs - self.inflates[batch_nodes].abs())

            # Reduce each individual's nodes via smooth union
            indiv_sdfs = torch.empty((len(batch_node_lists), V), device=self.device, dtype=self.dtype)
            offset = 0
            for b_i, nodes in enumerate(batch_node_lists):
                count = nodes.numel()
                sdf = signed_sdfs[offset : offset + 1]  # (1, V)
                for k in range(1, count):
                    g_idx = nodes[k]
                    s = self.signs[g_idx]
                    sdf = s * sdUnion(
                        s * sdf,
                        s * signed_sdfs[offset + k : offset + k + 1],
                        self.smoothings[g_idx],
                    )
                indiv_sdfs[b_i] = sdf.squeeze(0)
                offset += count

            fitnesses[batch_indivs] = indiv_sdfs.square().mean(dim=-1)

        return fitnesses

    def gen_batches(self, max_nodes: int) -> list[torch.Tensor]:
        """
        Greedily packs similar-colored individuals into batches of at most `max_nodes` total nodes.
        """
        P = self.colors.shape[0]
        norm_colors = torch.nn.functional.normalize(self.colors, dim=-1)
        unassigned = torch.ones(P, device=self.device, dtype=torch.bool)
        batches: list[torch.Tensor] = []

        while unassigned.any():
            # Pick first unassigned individual as seed
            seed = int(torch.where(unassigned)[0][0].item())
            sim = norm_colors @ norm_colors[seed]  # (P,)

            batch: list[int] = []
            rem_nodes = max_nodes

            while True:
                eligible = unassigned & (self.node_counts <= rem_nodes)
                if not eligible.any():
                    # Guarantee progress if a single individual exceeds max_nodes
                    if not batch:
                        batch.append(seed)
                        unassigned[seed] = False
                    break

                masked_sim = torch.where(eligible, sim, -torch.inf)
                chosen = int(masked_sim.argmax().item())
                batch.append(chosen)
                unassigned[chosen] = False
                rem_nodes -= int(self.node_counts[chosen].item())

            batches.append(torch.tensor(batch, device=self.device, dtype=torch.int64))

        return batches