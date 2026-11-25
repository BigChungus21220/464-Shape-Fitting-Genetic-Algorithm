# crossover / mutation operators, gene definitions

from dataclasses import dataclass
from utils import quat_multiply
from collections.abc import Callable
from typing import cast, Any
from utils import quat_random, quat_apply
import torch
from sdf import sdUnion, sdBox, sdRingSegment, sdSegment, sdCappedCylinder, sdCappedCone, sdPoint
import inspect
from utils import unique_indices
import torch.nn.functional as F
import isoext as ix


SdfType = Callable[..., torch.Tensor]

# functions taking a (B, V, 3) tensor of sample points, and n_params (B, 1, 1) tensors of input params returning a (B, V) tensor
# function, n_params pairs
SDF_FUNCS: list[tuple[SdfType, int]] = [(func, len(inspect.signature(func).parameters) - 1) for func in [
    sdBox,
    sdRingSegment,
    sdSegment,
    sdCappedCylinder,
    sdCappedCone,
    sdPoint
]]
MAX_PARAMS = max(n for _, n in SDF_FUNCS)
COLOR_DIM = 8
COLOR_MUT = 0.01
GRAD_DESC_FIELDS = ("rotations", "translations", "inflates", "smoothings", "params")


@dataclass
class SDFNode(ix.sdf.SDFProtocol):
    func: SdfType
    id: torch.Tensor # (1,)
    rotation: torch.Tensor # (1, 1, 4)
    translation: torch.Tensor # (1, 1, 3)
    inflate: torch.Tensor # (1, 1)
    smoothing: torch.Tensor # (1, 1)
    sign: torch.Tensor # (1, 1)
    params: torch.Tensor # (1, C, 1, 1)
    
    def __call__(self, p: torch.Tensor):
        transformed = quat_apply(self.rotation, p) - self.translation
        return (self.func(transformed, *self.params.unbind(1)) - self.inflate.abs())
    
    def to(self, to: torch.device | torch.dtype | None, other: torch.device | torch.dtype | None = None):
        device = to if isinstance(to, torch.device) else (other if isinstance(other, torch.device) or other is None else None)
        dtype = to if isinstance(to, torch.dtype) else (other if isinstance(other, torch.dtype) or other is None else None)
        return SDFNode(
            self.func,
            self.id.to(device=device, dtype=dtype),
            self.rotation.to(device=device, dtype=dtype),
            self.translation.to(device=device, dtype=dtype),
            self.inflate.to(device=device, dtype=dtype),
            self.smoothing.to(device=device, dtype=dtype),
            self.sign.to(device=device, dtype=dtype),
            self.params.to(device=device, dtype=dtype)
        )

    def serialize(self):
        func_name = self.func.__name__
        return {
            "type": func_name,
            "id": self.id.tolist(),
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "inflate": self.inflate.tolist(),
            "smoothing": self.smoothing.tolist(),
            "sign": self.sign.tolist(),
            "params": self.params.tolist()
        }
    
    @classmethod
    def deserialize(cls, dict: dict[str, Any], device: torch.device | str, dtype: torch.dtype):
        func = next((func for func, n_params in SDF_FUNCS if func.__name__ == dict["type"]), None)
        if func is None:
            raise ValueError(f"No function matching type '{dict["type"]}'")
        dev = torch.device(device)
        
        id = torch.tensor(dict["id"], device=dev, dtype=dtype)
        rotation = torch.tensor(dict["rotation"], device=dev, dtype=dtype)
        translation = torch.tensor(dict["translation"], device=dev, dtype=dtype)
        inflate = torch.tensor(dict["inflate"], device=dev, dtype=dtype)
        smoothing = torch.tensor(dict["smoothing"], device=dev, dtype=dtype)
        sign = torch.tensor(dict["sign"], device=dev, dtype=dtype)
        params = torch.tensor(dict["params"], device=dev, dtype=dtype)
        
        return cls(func, id, rotation, translation, inflate, smoothing, sign, params)
        

@dataclass
class SDF(ix.sdf.SDFProtocol):
    nodes: list[SDFNode]
    color: torch.Tensor # (COLOR_DIM,)
    
    def __call__(self, p: torch.Tensor):
        """Evaluates the function at verts, (..., 3) -> (...)"""
        p_flat = p.flatten(end_dim=-2).unsqueeze(0)
        sdf = self.nodes[0].sign*self.nodes[0](p_flat)
        for i in range(1, len(self.nodes)):
            node = self.nodes[i]

            sdf = node.sign*sdUnion(node.sign*sdf, node(p_flat), node.smoothing)
       
        return sdf.reshape(p.shape[:-1])
    
    def to(self, to: torch.device | torch.dtype | None, other: torch.device | torch.dtype | None = None):
        device = to if isinstance(to, torch.device) else (other if isinstance(other, torch.device) or other is None else None)
        dtype = to if isinstance(to, torch.dtype) else (other if isinstance(other, torch.dtype) or other is None else None)
        new_color = self.color.to(device=device, dtype=dtype)
        new_nodes: list[SDFNode] = []
        for node in self.nodes:
            new_nodes.append(node.to(device, dtype))
            
        return SDF(new_nodes, new_color)
    
    def serialize(self):
        """Converts to a json serializable object"""
        return {
            "nodes": [node.serialize() for node in self.nodes],
            "color": self.color.tolist()
        }
    
    @classmethod
    def deserialize(cls, dict: dict[str, Any], device: torch.device | str, dtype: torch.dtype):
        """Converts from a json object to an SDF"""
        dev = torch.device(device)
        return cls(
            [SDFNode.deserialize(node_dict, device, dtype) for node_dict in dict["nodes"]],
            torch.tensor(dict["color"], device=dev, dtype=dtype)
        )



# ^ does that, but more and harder
class Population:
    # P: population size
    # N: population node count
    # C: number of color dimensions
    # V: number of model vertices
    # M: max param count
    
    device: torch.device
    dtype: torch.dtype
    
    population: list[torch.Tensor] # [P,] arr of tensor, node indices
    node_counts: torch.Tensor # (P,) int, node count of each individual
    node_to_indiv: torch.Tensor # (N,) int, map back to population from nodes
    colors: torch.Tensor # (P, C) float, individual "colors", tracks speciation

    func_ids: torch.Tensor # (N,) int, function id for each node
    ids: torch.Tensor # (N,) float, node ids, tracks unique genes
    rotations: torch.Tensor # (N, 1, 4) float, quaternion rotation applied to input points
    translations: torch.Tensor # (N, 1, 3) float, offset applied to input points
    inflates: torch.Tensor # (N, 1) float, offset applied to the zero of the sdf
    smoothings: torch.Tensor # (N, 1) float, smoothing factors to apply
    signs: torch.Tensor # (N, 1) float {-1, 1}, whether each node is additive or subtractive
    # `a (-) b = -(a (+) -b)`, where `a` and `b` are sdfs, and `(+)` and `(-)` are sdf union and sdf subtraction
    
    params: torch.Tensor # (N, M, 1, 1) params for each function (not all are used for every function type)
    
    
    def train(self, verts: torch.Tensor, normals: torch.Tensor, n_iter: int, max_nodes_per_batch: int, mut_p: float, mut_std: float, mut_max_angle: float, species_thresh: float, 
              do_fitness_sharing: bool, keep_best: int, callback: Callable[[int, int, int, int, torch.Tensor, torch.Tensor, torch.Tensor, list[SDF]], None], summary_count: int,
          refine_every: int = 10, refine_steps: int = 10, refine_lr: float = 1e-2):

        """Main training loop

        Args:
            verts (torch.Tensor): Target model to match (1, V, 3)
            normals (torch.Tensor): Target model to match (1, V, 3)
            n_iter (int): Max iterations
            max_nodes_per_batch (int): Maximum nodes per batch, only affects performance
            mut_p (float): Probability each node recieves mutation
            mut_std (float): Standard deviation of node mutation
            mut_max_angle (float): Maximum range of rotation mutation
            species_thresh (float): Threshold for color similarity for two individuals to be considered the same species
            do_fitness_sharing (bool): Whether to apply a fitness sharing correction
            keep_best (int): How many of the best individuals to keep in the population
            callback (Callable[[int, int, int, int, torch.Tensor, torch.Tensor, torch.Tensor | None, list[SDF]], None]): Callback function
            summary_count (int): Number of species to report on each time callback is called
        """
        
        P = len(self.population)
        
        species_labels = torch.arange(P, device=self.device, dtype=torch.int64) # (P,) species index for each individual
        species_colors = self.colors.clone() # (S, C), species color per species, sorted by fitness of best member
        species_reps = species_labels.clone() # (S,) population index of best member of each species
        
        for i in range(n_iter):
            if refine_every > 0 and i % refine_every == 0:
                before, after = self.grad_desc(verts, max_nodes_per_batch, refine_steps, refine_lr)
                print(f"Refine: mean loss {before:.5g} -> {after:.5g}")
        
            loss = self.evaluate(verts, max_nodes_per_batch).clamp_max(1e20)
            
            # ranking based fitness (maybe not optimal, research needed)
            order = torch.argsort(loss, descending=False, stable=True)
            fitness = torch.empty_like(loss)
            fitness[order] = torch.arange(P, 0, -1, device=loss.device, dtype=loss.dtype)
            
            # species tracking
            species_colors, species_labels, species_reps, species_counts = self.update_species(order, species_colors, species_thresh)
            
            # trigger callback
            callback(i+1, n_iter, P, species_reps.numel(), species_counts, loss[species_reps[:summary_count]], 
                     species_colors[:summary_count, :3].clip(0.0, 1.0), [self.get_individual(i) for i in species_reps[:summary_count]])
            
            # strategy to preserve diversity (suggested by LLM, research needed)
            if do_fitness_sharing:
                fitness /= species_counts[species_labels].to(self.dtype)
            
            keep = species_reps[:keep_best]
            self.crossover(fitness, keep)
            # kept individuals are placed at population[:|keep|]
            self.mutate(mut_p, mut_std, mut_max_angle, keep=torch.arange(keep.numel()))
    
    
    def update_species(self, fitness_order: torch.Tensor, colors: torch.Tensor, thresh: float):
        """Tracks species updates"""
        P = len(self.population)
        
        # match each individual to its nearest existing species
        dists = torch.cdist(colors, self.colors) # (S, P)
        species_distance, nearest_species = torch.min(dists, dim=0) # (P,), (P,)

        surviving_mask = species_distance < thresh
        new_idx = (~surviving_mask).nonzero(as_tuple=True)[0] # (N_new,)
        
        # compact surviving species IDs to contiguous [0, S_surv)
        surviving_old_ids = nearest_species[surviving_mask]
        _, compact_surviving_labels = torch.unique(surviving_old_ids, return_inverse=True)
        num_surviving = int(compact_surviving_labels.max().item()) + 1 if surviving_old_ids.numel() > 0 else 0

        species_labels = torch.empty(P, device=self.device, dtype=torch.int64)
        species_labels[surviving_mask] = compact_surviving_labels
        species_labels[new_idx] = num_surviving + torch.arange(
            new_idx.numel(), device=self.device, dtype=torch.int64
        )

        num_species = num_surviving + new_idx.numel()
        species_counts = torch.bincount(species_labels, minlength=num_species)

        # update species colors to the fittest member of each species
        ordered_species = species_labels[fitness_order]  # (P,)

        # Find the first occurrence of each species along `order`
        first_pos_in_order = torch.full((num_species,), P, device=self.device, dtype=torch.int64)
        first_pos_in_order.scatter_reduce_(0, ordered_species, torch.arange(P, device=self.device), reduce='amin')
        
        sorted_first_pos, old_to_new_species = torch.sort(first_pos_in_order)
        inv_map = torch.empty_like(old_to_new_species)
        inv_map[old_to_new_species] = torch.arange(num_species, device=self.device)

        # Population indices of representatives, ordered by fitness descending:
        reps = fitness_order[sorted_first_pos]  # (S,)

        colors = self.colors[reps]
        counts = species_counts[old_to_new_species]
        labels = inv_map[species_labels]
        
        return colors, labels, reps, counts
    
    
    def get_individual(self, idx: int | torch.Tensor):
        """Returns an individual sdf"""
        nodes = []
        for node in self.population[idx]:
            func, n_params = SDF_FUNCS[self.func_ids[node]]
            node_slice = slice(node, node+1)
            nodes.append(SDFNode(
                func, 
                self.ids[node_slice], 
                self.rotations[node_slice], 
                self.translations[node_slice], 
                self.inflates[node_slice], 
                self.smoothings[node_slice], 
                self.signs[node_slice],
                self.params[node_slice, :n_params]
            ))
        
        return SDF(nodes, self.colors[idx])
        
    
    def __init__(self, 
            pop_size: int, 
            std: float,
            p_add: float, 
            min_nodes: int, 
            max_nodes: int, 
            func_weights: list[float], 
            device: torch.device | str, 
            dtype: torch.dtype
        ):
        
        self.dtype = dtype
        self.device = torch.device(device)
        weights = torch.tensor(func_weights, device=self.device, dtype=dtype)
        
        # figure out how many nodes each individual has
        self.node_counts = torch.randint(min_nodes, max_nodes + 1, (pop_size,), device=self.device, dtype=torch.int64)
        counts_cpu = self.node_counts.tolist()
        self.node_to_indiv = torch.repeat_interleave(
            torch.arange(pop_size, device=self.device, dtype=torch.int64),
            self.node_counts,
        )
        N = self.node_to_indiv.shape[0]
        
        self.population = list(torch.split(torch.arange(N, device=self.device), counts_cpu))
        self.colors = torch.randn((pop_size, COLOR_DIM), device=self.device, dtype=self.dtype)
        
        # generate per node stuff
        self.func_ids = torch.multinomial(weights, N, replacement=True)
        self.ids = torch.rand((N,), device=self.device, dtype=self.dtype)
        self.rotations = quat_random(N, torch.pi, device=self.device, dtype=self.dtype)
        self.translations = torch.normal(0.0, std, (N, 1, 3), device=self.device, dtype=self.dtype)
        self.inflates = torch.normal(0.0, std, (N, 1), device=self.device, dtype=self.dtype)
        self.smoothings = torch.normal(0.0, std,(N, 1), device=self.device, dtype=self.dtype)
        self.signs = torch.where(torch.rand((N, 1)) < p_add, 1.0, -1.0).to(self.dtype).to(self.device)
        self.params = torch.normal(0.0, std, (N, MAX_PARAMS, 1, 1), device=self.device, dtype=self.dtype)


    def evaluate(self, verts: torch.Tensor, max_nodes_per_batch: int, backward: bool = False, batches=None):
        """
        Returns the loss of each individual on input verts
        (1, V, 3) -> (P,)
        """
        # This function has 8 indexing levels:
        # Global individual indexing (P)
        # Global node indexing (N)
        # Batch individual indexing (B)
        # Active batch individual indexing at step i (R)
        # Node position within an individual (I)
        # Batch node indexing (K)
        # Unique batch node indexing (U)
        # Unique batch node function indexing (H)
        
        V = verts.shape[1]
        losses = torch.empty((len(self.population),), device=self.device, dtype=self.dtype)

        if batches is None:
            batches = self.gen_batches(max_nodes_per_batch)
        
        for indiv_idx in batches: # B -> P
            indivs = [self.population[i] for i in indiv_idx.tolist()] # B -> (I -> N)
            
            bnodes = torch.cat(indivs) # K -> N
            bcounts = torch.tensor([len(indiv) for indiv in indivs], device=self.device, dtype=torch.int64) # (B,)
            boffsets = torch.cumsum(F.pad(bcounts[:-1], (1, 0)), dim=0) # B -> K
            max_bcount = cast(int, bcounts.max().item())
            
            unique_to_bnodes, bnodes_to_unique = unique_indices(self.ids[bnodes]) # U -> K, K -> U
            unique_to_nodes = bnodes[unique_to_bnodes] # U -> N
            U = unique_to_bnodes.numel()
            
            func_ids = self.func_ids[unique_to_nodes] # (U,)
            
            transformed = quat_apply(self.rotations[unique_to_nodes], verts) - self.translations[unique_to_nodes] # (U, V, 3)
            
            node_sdfs = torch.empty((U, V), device=self.device, dtype=self.dtype) # (U, V)
            for f_id, (func, n_params) in enumerate(SDF_FUNCS):
                local_idx = (func_ids == f_id).nonzero(as_tuple=True)[0] # H -> U
                if local_idx.numel() == 0:
                    continue
                
                node_idx = unique_to_nodes[local_idx] # H -> N
                params = self.params[node_idx, :n_params].unbind(1) # n_params x (H, 1, 1)
                samples = transformed[local_idx] # (H, V, 3)
                node_sdfs[local_idx] = func(samples, *params) # (H, V)
            
            inflated_sdfs = node_sdfs - self.inflates[unique_to_nodes].abs()
            
            # step over all indiv nodes at once
            indiv_sdfs = inflated_sdfs[bnodes_to_unique[boffsets]] # (B, V)
            for i in range(1, max_bcount):
                valid_idx = (i < bcounts).nonzero(as_tuple=True)[0] # B -> R
                node_idx = bnodes[boffsets[valid_idx] + i] # R -> N
                s = self.signs[node_idx]
                k = self.smoothings[node_idx]
                sdf = inflated_sdfs[bnodes_to_unique[boffsets[valid_idx] + i]] # (R, V)
                indiv_sdfs[valid_idx] = s*sdUnion(s*indiv_sdfs[valid_idx], sdf, k)

            batch_loss = torch.sum(torch.square(indiv_sdfs), dim=1)*(1 + 0.1*self.node_counts[indiv_idx])
            if backward:
                # differentiate batch_loss using chain rule
                batch_loss.sum().backward()
            # take losses, detach the tracked operations
            losses[indiv_idx] = batch_loss.detach() if backward else batch_loss
            
        return losses

    def grad_desc(self, verts: torch.Tensor, max_nodes_per_batch: int, steps: int = 10, lr: float = 1e-2):
        """
        Adam steps on the continuous tensors (that the names in GRAD_DESC_FIELDS point to on self), 
        each candidate keeps the result only if its loss improved.
        Returns (mean loss before, mean loss after) over the population.
        """
        N = self.func_ids.shape[0]
        self.ids = torch.arange(N, device=self.device, dtype=self.dtype) # unique ids: no merged copies
        batches = self.gen_batches(max_nodes_per_batch)

        saved = {n: getattr(self, n).detach().clone() for n in GRAD_DESC_FIELDS}
        tensors = [getattr(self, n) for n in GRAD_DESC_FIELDS]

        # track operations on tensors in GRAD_DESC_FIELDS
        for t in tensors:
            t.requires_grad_(True)
        opt = torch.optim.Adam(tensors, lr=lr)

        start_losses = None
        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            losses = self.evaluate(verts, max_nodes_per_batch, backward=True, batches=batches)
            # keep track of the loss before running grad desc to rollback to if it steps in worse direction
            if step == 0:
                start_losses = losses.clone()
            for t in tensors:
                if t.grad is not None:
                    t.grad.nan_to_num_(0.0, posinf=0.0, neginf=0.0)
            # use Adam's rules to determine step size and takes the 'step'
            opt.step()
            with torch.no_grad():
                self.rotations.copy_(F.normalize(self.rotations, p=2, dim=-1))

        for t in tensors:
            t.requires_grad_(False)
            t.grad = None

        # accept-if-better, per candidate
        with torch.no_grad():
            final = self.evaluate(verts, max_nodes_per_batch, batches=batches)
            worse = ~(final <= start_losses)            # NaN counts as worse
            node_worse = worse[self.node_to_indiv]      # (N,)
            for n in GRAD_DESC_FIELDS:
                getattr(self, n)[node_worse] = saved[n][node_worse]
            kept = torch.where(worse, start_losses, final)

        print(f"  best {start_losses.min().item():.5g} -> {kept.min().item():.5g}, "
            f"reverted {int(worse.sum())}/{worse.numel()}")
        return start_losses.mean().item(), kept.mean().item()

    def refine_top(self, verts: torch.Tensor, max_nodes_per_batch: int = 256, top_k: int = 5, steps: int = 300, lr: float = 1e-2):
        """Keeps only the top_k lowest-loss individuals and runs deep gradient descent on them."""
        # find top_k candidates with lowest loss
        with torch.no_grad():
            losses = self.evaluate(verts, max_nodes_per_batch)
            top_indivs = torch.topk(losses, k=min(top_k, losses.numel()), largest=False).indices

        # create bool mask for nodes belonging to top_indivs
        node_mask = torch.isin(self.node_to_indiv, top_indivs)
        for attr in ["func_ids", "signs", "node_to_indiv", "ids", *GRAD_DESC_FIELDS]:
            if hasattr(self, attr):
                setattr(self, attr, getattr(self, attr)[node_mask])

        # Remap self.population and node indices to the newly sliced arrays (0 .. N_kept - 1)
        old_to_new = torch.full((node_mask.numel(),), -1, device=self.device, dtype=torch.int64)
        old_to_new[node_mask] = torch.arange(int(node_mask.sum()), device=self.device)
        self.population = [old_to_new[self.population[i]] for i in top_indivs.tolist()]

        if hasattr(self, "colors"):
            self.colors = self.colors[top_indivs]

        indiv_map = torch.full((losses.numel(),), -1, device=self.device, dtype=torch.int64)
        indiv_map[top_indivs] = torch.arange(top_indivs.numel(), device=self.device)
        self.node_to_indiv = indiv_map[self.node_to_indiv]

        stats = self.grad_desc(verts, max_nodes_per_batch, steps=steps, lr=lr)

        # Re-order self.population so index 0 is guaranteed to be the best
        with torch.no_grad():
            final_losses = self.evaluate(verts, max_nodes_per_batch)
            order = torch.argsort(final_losses, descending=False)
            self.population = [self.population[i] for i in order.tolist()]
            if hasattr(self, "colors"):
                self.colors = self.colors[order]

        return stats
        
        
    def crossover(self, weights: torch.Tensor, keep: torch.Tensor) -> None:
        """
        Applies crossover and preserves specified individuals at the start of the population.
        Args:
            weights (torch.Tensor): Per-individual selection weights (P,)
            keep (torch.Tensor): Indices of individuals to preserve at slots 0..K-1 (K,)
        """
        P = self.node_counts.shape[0]
        keep = keep.to(device=self.device, dtype=torch.int64)
        K = keep.numel()
        if K > P:
            raise ValueError(f"Cannot keep {K} individuals in a population of size {P}.")

        old_offsets = torch.cumsum(F.pad(self.node_counts[:-1], (1, 0)), dim=0)  # (P,)

        def expand_indivs(
            indiv_indices: torch.Tensor, child_offset: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
            n_indivs = indiv_indices.shape[0]
            if n_indivs == 0:
                empty = torch.empty(0, device=self.device, dtype=torch.int64)
                return empty, empty

            counts = self.node_counts[indiv_indices]
            total = int(counts.sum().item())
            local_child = torch.repeat_interleave(
                torch.arange(n_indivs, device=self.device, dtype=torch.int64), counts
            )
            local_offsets = torch.cumsum(F.pad(counts[:-1], (1, 0)), dim=0)
            local_pos = torch.arange(
                total, device=self.device, dtype=torch.int64
            ) - torch.repeat_interleave(local_offsets, counts)
            old_node_idx = old_offsets[indiv_indices][local_child] + local_pos
            return local_child + child_offset, old_node_idx

        # 1. Expand all preserved individuals into child slots 0 .. K - 1
        elite_children, elite_nodes = expand_indivs(keep, child_offset=0)
        num_children = P - K

        if num_children == 0:
            self.colors = self.colors[keep]
            gather_idx = elite_nodes
            new_node_to_indiv = elite_children
        else:
            # 2. Sample parent pairs for the remaining P - K slots (child indices K .. P - 1)
            a = torch.multinomial(weights, num_samples=num_children, replacement=True)
            b = torch.multinomial(weights, num_samples=num_children, replacement=True)
            self.colors = torch.cat(
                [
                    self.colors[keep],
                    0.5 * (self.colors[a] + self.colors[b]),
                ],
                dim=0,
            )

            child_a, nodes_a = expand_indivs(a, child_offset=K)
            keep_a = torch.rand(nodes_a.shape[0], device=self.device) < 0.5
            child_a, nodes_a = child_a[keep_a], nodes_a[keep_a]

            child_b, nodes_b = expand_indivs(b, child_offset=K)
            keep_b = torch.rand(nodes_b.shape[0], device=self.device) < 0.5
            child_b, nodes_b = child_b[keep_b], nodes_b[keep_b]

            # 3. Combine preserved nodes (keys < 2*K) with crossover nodes (keys >= 2*K)
            keys = torch.cat([elite_children * 2, child_a * 2, child_b * 2 + 1])
            selected_nodes = torch.cat([elite_nodes, nodes_a, nodes_b])
            selected_children = torch.cat([elite_children, child_a, child_b])

            # Guarantee every crossover child (indices >= K) has at least 1 node
            counts_check = torch.bincount(selected_children, minlength=P)
            empty_children = (counts_check == 0).nonzero(as_tuple=True)[0]
            if empty_children.numel() > 0:
                fallback_nodes = old_offsets[a[empty_children - K]]
                keys = torch.cat([keys, empty_children * 2])
                selected_nodes = torch.cat([selected_nodes, fallback_nodes])
                selected_children = torch.cat([selected_children, empty_children])

            order = torch.argsort(keys, stable=True)
            gather_idx = selected_nodes[order]
            new_node_to_indiv = selected_children[order]

        # 4. Rebuild structural mappings and gather all per-node tensors
        new_node_counts = torch.bincount(new_node_to_indiv, minlength=P)
        N_new = gather_idx.shape[0]

        self.node_counts = new_node_counts
        self.node_to_indiv = new_node_to_indiv
        self.population = list(
            torch.split(
                torch.arange(N_new, device=self.device, dtype=torch.int64),
                new_node_counts.tolist(),
            )
        )

        self.func_ids = self.func_ids[gather_idx]
        self.ids = self.ids[gather_idx]
        self.rotations = self.rotations[gather_idx]
        self.translations = self.translations[gather_idx]
        self.inflates = self.inflates[gather_idx]
        self.smoothings = self.smoothings[gather_idx]
        self.signs = self.signs[gather_idx]
        self.params = self.params[gather_idx]
    
    
    def mutate(self, p_mut: float, std: float, max_angle: float, keep: torch.Tensor):
        """
        Mutates all nodes with probability p_mut
        Args:
            p_mut (float): Probability each node mutates
            std (float): Standard deviation of mutation magnitude
            max_angle (float): Maximum change in rotation angle
            keep (Tensor): Individuals which may not be mutated
        """
        # Since mutation occurs per node, there is an implicit bias towards shorter solutions
        
        mask = torch.rand(self.func_ids.shape, device=self.device) < p_mut
        if keep.numel() > 0:
            keep_nodes = torch.cat(self.population[:keep.numel()])
            mask[keep_nodes] = False
        
        idx = mask.nonzero(as_tuple=True)[0]
        K = idx.numel()
        
        # get the indices of pop which have changed
        back_idx = torch.unique(self.node_to_indiv[idx])
        D = back_idx.shape[0]
        self.colors[back_idx] += F.normalize(torch.normal(0.0, COLOR_MUT, (D, COLOR_DIM), device=self.device, dtype=self.dtype), p=2, dim=-1)
        
        self.ids[idx] = torch.rand((K,), device=self.device, dtype=self.dtype)
        
        r_offsets = quat_random(K, max_angle, device=self.device, dtype=self.dtype)
        self.rotations[idx] = F.normalize(quat_multiply(self.rotations[idx], r_offsets), p=2, dim=-1)
        
        self.translations[idx] += torch.normal(0.0, std, (K, 1, 3), device=self.device, dtype=self.dtype)
        self.inflates[idx] += torch.normal(0.0, std, (K, 1), device=self.device, dtype=self.dtype)
        self.smoothings[idx] += torch.normal(0.0, std,(K, 1), device=self.device, dtype=self.dtype)
        self.params[idx] += torch.normal(0.0, std, (K, MAX_PARAMS, 1, 1), device=self.device, dtype=self.dtype)
    
    
    def gen_batches(self, k: int):
        """
        Greedily packs individuals into batches of at most `k` unique nodes,
        maximizing shared nodes and breaking ties by minimum combined function types.
        """
        
        P = len(self.population)
        ids_cpu = self.ids.cpu().tolist()
        funcs_cpu = self.func_ids.cpu().tolist()

        pop_ids: list[set[float]] = []
        pop_funcs: list[set[int]] = []
        for indiv in self.population:
            node_indices = indiv.tolist()
            pop_ids.append({ids_cpu[idx] for idx in node_indices})
            pop_funcs.append({funcs_cpu[idx] for idx in node_indices})

        unassigned = set(range(P))
        batches: list[torch.Tensor] = []

        while unassigned:
            # Start the batch with the unassigned individual that has the fewest function types
            seed = min(unassigned, key=lambda i: len(pop_funcs[i]))
            unassigned.remove(seed)

            batch = [seed]
            group_ids = set(pop_ids[seed])
            group_funcs = set(pop_funcs[seed])

            while len(group_ids) < k and unassigned:
                best_cand: int | None = None
                best_key: tuple[int, int] | None = None

                for cand in unassigned:
                    cand_ids = pop_ids[cand]
                    new_nodes = len(cand_ids - group_ids)
                    if len(group_ids) + new_nodes > k:
                        continue

                    shared_nodes = len(cand_ids) - new_nodes
                    combined_types = len(group_funcs | pop_funcs[cand])
                    key = (-shared_nodes, combined_types)

                    if best_key is None or key < best_key:
                        best_key = key
                        best_cand = cand

                if best_cand is None:
                    break

                batch.append(best_cand)
                unassigned.remove(best_cand)
                group_ids.update(pop_ids[best_cand])
                group_funcs.update(pop_funcs[best_cand])

            batches.append(torch.tensor(batch, device=self.device, dtype=torch.int64))

        return batches