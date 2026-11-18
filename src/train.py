# main genetic algorithm loop

import torch
from sdf import SdfNode
from model import generate
import trimesh
from model import evaluate, crossover, mutate
import random

mesh: trimesh.Trimesh = trimesh.load('datasets/shapes/sofa.obj', force='mesh') # type: ignore

# todo:
# each iteration, group nodes by type, then run them all batched (don't run duplicates)
#   give each node an id which changes on mutation or node crossover to track duplicates efficently
#   will require tons of memory, so more thought is required on how to handle batching
#     maybe use the color value to batch by species to maximize node overlap, making small batches more effective
# store a color value at the individual level to track speciation (crossover with average, mutate with random offset
#   use more than 3 channels to improve distinguishability)
#   then you can project back to 3 colors to get a viewable color via PCA



POP_SIZE = 1000
EPOCHS = 10000
mutation_chance = 0.01

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32

population: list[list[SdfNode]] = [generate(3, 20, device, dtype) for i in range(POP_SIZE)]
fitness = torch.zeros(POP_SIZE, device=device, dtype=dtype)
verts = torch.tensor(mesh.vertices, device=device, dtype=dtype).unsqueeze(0)

for epoch in range(EPOCHS):
    for i in range(POP_SIZE):
        fitness[i] = evaluate(population[i], verts)[0].clamp_max(1e20)
        
    best_idx = torch.argmin(fitness)
    
    new_population = []
    new_population.append(population[best_idx])
    
    for i in range(POP_SIZE - 1):
        p1, p2 = random.choices(population, weights=(torch.max(fitness) - fitness).cpu().tolist(), k=2)
        child = crossover(p1, p2)
        mutate(child, mutation_chance)
        new_population.append(child)
        
    population = new_population
    
    print(fitness.topk(15, largest=False))