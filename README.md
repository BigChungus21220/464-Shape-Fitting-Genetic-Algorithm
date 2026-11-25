A genetic algorithm for shape fitting

## Setup

Use python `3.12` or later

Create a venv: `py -m venv .venv`

Activate the venv: `.venv/Scripts/activate`

Install the requirements: `pip install -r requirements.txt`

Run the training `python src/main.py`

## Usage

```
python src/main.py --shape dog -n 300 -p 1000
```

Run `python src/main.py --help` for the full list. Common flags:

| Flag | Default | Meaning |
|---|---|---|
| `--shape` | `dog` | Target mesh: `dog`, `hand`, `pot`, `rod`, `sofa`, or a path to any mesh file |
| `-n`, `--generations` | `300` | Number of generations (epochs). Each generation evaluates, selects, crosses over and mutates the whole population |
| `-p`, `--pop-size` | `1000` | Number of individuals. Larger finds better fits but each generation is slower and uses more GPU memory |
| `--keep-best` | `2` | Elitism: the best individuals of the top species are copied unchanged into the next generation |
| `--mut-p` | `0.01` | Probability that each primitive (node) gets mutated |
| `--mut-std` | `0.01` | Size of the mutation noise added to parameters, translations, inflates and smoothings |
| `--species-thresh` | `0.5` | Color distance under which two individuals count as the same species |
| `--no-fitness-sharing` | off | Stop dividing fitness by species size. Sharing keeps diversity; turning it off can cause a mass extinction |
| `--out` | `output/loss_history.png` | Where the loss plot is saved (updated every `--plot-every` generations) |
| `--seed` | random | Random seed, for reproducible runs |

## TODO
- Compare reconstruction with other metrics
  - try using true contour distance as loss
  - add eikonal constraint (normal alignment loss, like in assignment 1)
- research into GA selection strategies and decide on which we should be using
- implement gradient descent updates
  - Apply several steps of gradient descent to the population every few generations
  - At the end, run deeper gradient descent on the best individual for the final model
- improve loss function
  - apply eikonal loss
  - add some regularization terms
    - k should be small
    - node count should be low
  - occupancy loss

## Current strategy

### Fitness Function:

Given a pointcloud with points $p_i$ and normals $n_i$, the fitness of an individual is

$`\text{Fitness}_j = \text{Rank}_j\left(L\right)`$

$`L_j = \sum\left(\text{sdf}_j\left(p_i\right)^2 + \lambda\left(1 - n_i \cdot \nabla\text{sdf}_j\left(p_i\right)\right)^2 \right)`$

Where the sdf of an individual is defined $`\text{sdf}_j = \text{sdf}_{j, C_j}`$

$`\text{sdf}_{j, 1}\left(p_i\right) = s_{j,t} P_{j,i+1}\left(p_i\right)`$

$`\text{sdf}_{j, t+1}\left(p_i\right) = s_{j,t} \text{smin}(\text{sdf}_{j,t}\left(p_i\right), s_{j,t}P_{j,i+1}\left(p_i\right), k_{j, t+1})`$

- $`P_{j, t}: \mathbb{R}^3 \rightarrow \mathbb{R}`$: a primitive sdf function

- $`s_{j,t} \in \{-1, 1\}`$: the sign of the primitive solid

- $`k_{j,t} \in \mathbb{R}`$: the smoothing parameter

- $`C_{j} \in \mathbb{N}`$: the number of genes defined for this individual


### Genes:

Each individual, $j$ has $C_j$ genes:
1. $`P_{j, t}`$: primitive function
2. $`s_{j, t}`$: primitive sign
3. $`X_{j, t}`$: primitive specific parameters
4. $`q_{j, t}`$: rotation
5. $`o_{j, t}`$: translation
6. $`d_{j, t}`$: inflate
7. $`k_{j, t}`$: smoothing


Mutation:
 - Add a random amount to a $X_{j, t}$, $q_{j, t}$, $o_{j, t}$, $d_{j, t}$, or $k_{j, t}$

Crossover:
 - Each parent selects each of its genes with a 50% chance to pass of to the child

Selection:
 - probability of selection is proportional to ranking


### Training:
 - use keep best + weighted choice


## Parameter Search & Ablasion

### Parameter Search
 - Run a grid search over population, mutation rate(s)
 - Save the best result to a table, then apply a blur in population, mutation rate space to reduce noise

### Ablasion
 - Test the model with an eikonal constraint
 - Test the model with subsets of the primatives rather than the full set
 - Test the model without smooth boolean operations
 - Test with intermediate gradient descent hybrid approach


## Reading List

Not nessecarily useful

https://doi.org/10.1016/j.cad.2016.01.001
 - points -> segment -> fit via GA
 - explict reconstruction length penalty
https://doi.org/10.1007/978-3-030-94893-1_3
https://doi.org/10.1109/cvpr.2018.00578