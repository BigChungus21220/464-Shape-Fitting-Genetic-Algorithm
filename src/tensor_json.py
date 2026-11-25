"""Round-trip a PyTorch tensor through JSON.

JSON layout:
    {"dtype": "float32", "shape": [2, 3], "data": [ ...flat list... ]}

Handles: any shape (incl. scalars/empty), int/bool/float/bfloat16/float16,
complex64/128, and NaN/inf (stored as the strings "nan", "inf", "-inf",
since they are not valid JSON numbers).
"""
import json
import math

import torch

_COMPLEX_TO_REAL = {
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}


def tensor_to_dict(t: torch.Tensor) -> dict:
    t = t.detach().cpu()
    dtype = str(t.dtype).replace("torch.", "")
    shape = list(t.shape)

    if t.is_complex():
        t = torch.view_as_real(t.contiguous())  # adds a trailing dim of size 2

    flat = t.reshape(-1).tolist()
    if t.is_floating_point():
        flat = [x if math.isfinite(x) else str(x) for x in flat]

    return {"dtype": dtype, "shape": shape, "data": flat}


def dict_to_tensor(d: dict) -> torch.Tensor:
    dtype = getattr(torch, d["dtype"])
    shape = d["shape"]
    flat = [float(x) if isinstance(x, str) else x for x in d["data"]]

    if dtype.is_complex:
        real = torch.tensor(flat, dtype=_COMPLEX_TO_REAL[dtype])
        return torch.view_as_complex(real.reshape(*shape, 2).contiguous())

    return torch.tensor(flat, dtype=dtype).reshape(shape)


def save_tensor(t: torch.Tensor, path: str, indent=None) -> None:
    with open(path, "w") as f:
        json.dump(tensor_to_dict(t), f, indent=indent)


def load_tensor(path: str) -> torch.Tensor:
    with open(path) as f:
        return dict_to_tensor(json.load(f))


if __name__ == "__main__":
    original = torch.randn(2, 3, 4)
    original[0, 0, 0] = float("nan")
    original[1, 2, 3] = float("-inf")

    save_tensor(original, "tensor.json")
    restored = load_tensor("tensor.json")

    assert restored.dtype == original.dtype
    assert restored.shape == original.shape
    assert torch.equal(restored.nan_to_num(), original.nan_to_num())
    print("Round trip OK:", restored.shape, restored.dtype)