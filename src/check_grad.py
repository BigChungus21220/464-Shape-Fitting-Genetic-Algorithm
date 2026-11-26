import torch
from model import Population, GRAD_DESC_FIELDS

def run_grad_check():
    torch.manual_seed(0)
    pop = Population(
        pop_size=4, std=1.0, p_add=0.8, min_nodes=3, max_nodes=5,
        func_weights=[1.0] * 6, device="cpu", dtype=torch.float64,
    )
    verts = torch.randn(1, 200, 3, dtype=torch.float64)
    pop.ids = torch.arange(pop.func_ids.shape[0], dtype=torch.float64)

    for n in GRAD_DESC_FIELDS:
        getattr(pop, n).requires_grad_(True)
    pop.evaluate(verts, pop.gen_batches(256), backward=True)
    auto = {n: getattr(pop, n).grad for n in GRAD_DESC_FIELDS}

    @torch.no_grad()
    def total_loss():
        return pop.evaluate(verts, pop.gen_batches(256)).sum().item()

    eps = 1e-6
    for n in GRAD_DESC_FIELDS:
        t, g = getattr(pop, n), auto[n]
        assert g is not None, f"{n} has no gradient"
        for _ in range(6):
            idx = tuple(int(torch.randint(s, (1,)).item()) for s in t.shape)
            old = t.data[idx].item()
            t.data[idx] = old + eps; lp = total_loss()
            t.data[idx] = old - eps; lm = total_loss()
            t.data[idx] = old
            fd = (lp - lm) / (2 * eps)
            assert torch.isclose(g[idx], torch.tensor(fd, dtype=torch.float64), rtol=1e-4, atol=1e-5), (
                f"Mismatch on {n}{idx}: autograd={g[idx].item()} vs fd={fd}"
            )
            print(f"{n:13s} {str(idx):18s} autograd={g[idx].item():+.4e}  fd={fd:+.4e}")

    before, after = pop.grad_desc(verts,normals = torch.zeros_like(verts), max_nodes_per_batch=256, steps=50, lr=1e-2)
    assert after < before, f"Expected loss to decrease, got {before} -> {after}"
    print(f"All gradient checks passed (mean loss: {before:.4f} -> {after:.4f}).")


if __name__ == "__main__":
    run_grad_check()