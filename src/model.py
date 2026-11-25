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
COLOR_MUT = 0.001



@dataclass
class SDFNode:
    func: SdfType
    id: torch.Tensor # (1,)
    rotation: torch.Tensor # (1, 1, 4)
    translation: torch.Tensor # (1, 1, 3)
    inflate: torch.Tensor # (1, 1)
    smoothing: torch.Tensor # (1, 1)
    sign: torch.Tensor # (1, 1)
    params: torch.Tensor # (1, C, 1, 1)
    
    def evaluate(self, verts: torch.Tensor):
        transformed = quat_apply(self.rotation, verts) - self.translation
        return (self.func(transformed, self.params.unbind(1)) - self.inflate)
    
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
class SDF:
    nodes: list[SDFNode]
    color: torch.Tensor # (COLOR_DIM,)
    idx: int
    
    def __call__(self, verts: torch.Tensor):
        sdf = self.nodes[0].evaluate(verts)
        for i in range(1, len(self.nodes)):
            node = self.nodes[i]
            sdf = node.sign*sdUnion(sdf, node.sign*node.evaluate(verts), node.smoothing)
        return sdf
            
    def serialize(self):
        return {
            "nodes": [node.serialize() for node in self.nodes],
            "color": self.color.tolist(),
            "idx": self.idx
        }
    
    @classmethod
    def deserialize(cls, dict: dict[str, Any], device: torch.device | str, dtype: torch.dtype):
        dev = torch.device(device)
        return cls(
            [SDFNode.deserialize(node_dict, device, dtype) for node_dict in dict["nodes"]],
            torch.tensor(dict["color"], device=dev, dtype=dtype),
            dict["idx"]
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
    ids: torch.Tensor # (N,) int, node ids, tracks unique genes
    rotations: torch.Tensor # (N, 1, 4) float, quaternion rotation applied to input points
    translations: torch.Tensor # (N, 1, 3) float, offset applied to input points
    inflates: torch.Tensor # (N, 1) float, offset applied to the zero of the sdf
    smoothings: torch.Tensor # (N, 1) float, smoothing factors to apply
    signs: torch.Tensor # (N, 1) float {-1, 1}, whether each node is additive or subtractive
    # `a (-) b = -(a (+) -b)`, where `a` and `b` are sdfs, and `(+)` and `(-)` are sdf union and sdf subtraction
    
    params: torch.Tensor # (N, M, 1, 1) params for each function (not all are used for every function type)
    
    
    def train(self, verts: torch.Tensor, n_iter: int, max_nodes_per_batch: int, mut_p: float, mut_std: float, mut_max_angle: float, callback: Callable[[int, int, torch.Tensor, torch.Tensor | None, list[SDF]], None], summary_count: int):
        P = len(self.population)
        projection = None
        for i in range(n_iter):
            loss = self.evaluate(verts, max_nodes_per_batch).clamp_max(1e20)
            
            order = torch.argsort(loss, descending=False, stable=True)  # lowest loss first
            fitness = torch.empty_like(loss)
            fitness[order] = torch.arange(P, 0, -1, device=loss.device, dtype=loss.dtype)
            
            best_loss, best_idx = loss.topk(summary_count, largest=False)
            
            best = [self.get_individual(idx) for idx in best_idx]
            
            self.crossover(fitness) # top individual is at population[0]
            self.mutate(mut_p, mut_std, mut_max_angle, keep=self.population[0])
            
            # wait for stuff to settle before choosing a color scheme
            if i+1 == 10:
                projection = self.get_color_projection()
            if projection is not None:
                callback(i+1, n_iter, best_loss, self.get_colors(*projection)[best_idx], best)
            else:
                callback(i+1, n_iter, best_loss, None, best)
                
            
    def get_color_projection(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the axes to project to rgb
        """
        P, C = self.colors.shape

        mean = self.colors.mean(dim=0, keepdim=True)
        centered = self.colors - mean

        _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
        components = cast(torch.Tensor, Vh[:3].T)  # (C, 3)

        max_abs_idx = components.abs().argmax(dim=0)
        signs = torch.sign(components[max_abs_idx, torch.arange(3, device=self.colors.device)])
        components = components * torch.where(signs == 0, 1.0, signs)
        return components, mean
    
    
    def get_colors(self, components: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        """
        Projects color genes to RGB values in [0, 1] using PCA.
        """
        
        centered = self.colors - mean
        rgb = centered @ components  # (P, 3)
        rgb_min = rgb.amin(dim=0, keepdim=True)
        rgb_max = rgb.amax(dim=0, keepdim=True)
        rgb = (rgb - rgb_min) / (rgb_max - rgb_min).clip(0.0, 1.0)

        return rgb
    
    
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


        #idx_int = idx
        if type(idx) is torch.Tensor:
            #print(f"idx: {idx.shape}")
            idx_int = int(idx.item())
        else:
            idx_int = idx
        
        return SDF(nodes, self.colors[idx], idx_int)
        
    
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


    def evaluate(self, verts: torch.Tensor, max_nodes_per_batch: int):
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
        fitnesses = torch.empty((len(self.population),), device=self.device, dtype=self.dtype)
        
        for indiv_idx in self.gen_batches(max_nodes_per_batch): # B -> P
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
            
            signed_sdfs = self.signs[unique_to_nodes]*(node_sdfs - self.inflates[unique_to_nodes].abs())
            
            # step over all indiv nodes at once (probably faster at the end of the day, idk)
            indiv_sdfs = signed_sdfs[bnodes_to_unique[boffsets]] # (B, V)
            for i in range(1, max_bcount):
                valid_idx = (i < bcounts).nonzero(as_tuple=True)[0] # B -> R
                node_idx = bnodes[boffsets[valid_idx] + i] # R -> N
                s = self.signs[node_idx]
                k = self.smoothings[node_idx]
                sdf = signed_sdfs[bnodes_to_unique[boffsets[valid_idx] + i]] # (R, V)
                indiv_sdfs[valid_idx] = s*sdUnion(indiv_sdfs[valid_idx], sdf, k)

            # take fitnesses
            fitnesses[indiv_idx] = torch.sum(torch.square(indiv_sdfs), dim=1)
            
        return fitnesses
        
        
    def crossover(self, weights: torch.Tensor) -> None:
        """
        Applies crossover and keep best population update
        Args:
            weights (torch.Tensor): Per individual weights (fitness)
        """
        P = self.node_counts.shape[0]
        old_offsets = torch.cumsum(F.pad(self.node_counts[:-1], (1, 0)), dim=0)  # (P,)
        best_idx = weights.argmax()

        # 1. Preserve the elite individual's exact nodes at child index 0
        elite_count = int(self.node_counts[best_idx].item())
        elite_nodes = (
            torch.arange(elite_count, device=self.device, dtype=torch.int64)
            + old_offsets[best_idx]
        )
        elite_children = torch.zeros(elite_count, device=self.device, dtype=torch.int64)
        elite_keys = torch.zeros(elite_count, device=self.device, dtype=torch.int64)

        if P == 1:
            self.colors = self.colors[best_idx : best_idx + 1]
            gather_idx = elite_nodes
            new_node_to_indiv = elite_children
        else:
            num_children = P - 1

            # 2. Sample parent pairs for the remaining P - 1 slots (child indices 1 .. P - 1)
            a = torch.multinomial(weights, num_samples=num_children, replacement=True)
            b = torch.multinomial(weights, num_samples=num_children, replacement=True)
            self.colors = torch.cat(
                [
                    self.colors[best_idx : best_idx + 1],
                    0.5 * (self.colors[a] + self.colors[b]),
                ],
                dim=0,
            )

            def sample_parent_nodes(parents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                counts = self.node_counts[parents]  # (P - 1,)
                total = int(counts.sum().item())
                # Shift child indices by +1 since index 0 is reserved for the elite individual
                child_idx = (
                    torch.repeat_interleave(
                        torch.arange(num_children, device=self.device, dtype=torch.int64),
                        counts,
                    )
                    + 1
                )
                local_offsets = torch.cumsum(F.pad(counts[:-1], (1, 0)), dim=0)
                local_pos = torch.arange(
                    total, device=self.device, dtype=torch.int64
                ) - torch.repeat_interleave(local_offsets, counts)
                old_node_idx = old_offsets[parents][child_idx - 1] + local_pos

                keep = torch.rand(total, device=self.device) < 0.5
                return child_idx[keep], old_node_idx[keep]

            child_a, nodes_a = sample_parent_nodes(a)
            child_b, nodes_b = sample_parent_nodes(b)

            # 3. Combine elite nodes (key 0) with crossover nodes (keys >= 2)
            keys = torch.cat([elite_keys, child_a * 2, child_b * 2 + 1])
            selected_nodes = torch.cat([elite_nodes, nodes_a, nodes_b])
            selected_children = torch.cat([elite_children, child_a, child_b])

            # Guarantee every crossover child (indices >= 1) has at least 1 node
            counts_check = torch.bincount(selected_children, minlength=P)
            empty_children = (counts_check == 0).nonzero(as_tuple=True)[0]
            if empty_children.numel() > 0:
                fallback_nodes = old_offsets[a[empty_children - 1]]
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
    
    
    def mutate(self, p_mut: float, std: float, max_angle: float, keep: torch.Tensor | None = None):
        """
        Mutates all nodes with probability p_mut
        Args:
            p_mut (float): Probability each node mutates
            std (float): Standard deviation of mutation magnitude
            max_angle (float): Maximum change in rotation angle
            keep (Tensor): Nodes indices which may not be mutate
        """
        # Since mutation occurs per node, there is an implicit bias towards shorter solutions
        
        mask = torch.empty(self.func_ids.shape, dtype=torch.bool).bernoulli_(p_mut)
        if keep is not None:
            mask[keep] = False
        
        idx = mask.nonzero(as_tuple=True)[0]
        K = idx.numel()
        
        back_idx = torch.unique(self.node_to_indiv[idx])
        D = back_idx.shape[0]
        
        # get the indices of pop which have changed
        self.colors[back_idx] += 0.01*F.normalize(torch.randn((D, COLOR_DIM), device=self.device, dtype=self.dtype), p=2, dim=-1)
        
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