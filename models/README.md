# Model Weights

The current local default points at the vendored DiffBrush copy under this repo:

```text
models/diffbrush/third_party_repo/model_zoo/DiffBrush-ckpt.pt
```

`handgen` calls DiffBrush in-process through `handgen/diffbrush/runner.py`, which builds the model once per `document render` and reuses it across every prose chunk and variable in that render.

If you want to use a different checkout or keep weights outside this repo, override the paths through:

```bash
export DIFFBRUSH_ROOT=/path/to/DiffBrush
export DIFFBRUSH_CHECKPOINT=/path/to/DiffBrush-ckpt.pt
export DIFFBRUSH_DEVICE=auto   # or 'cpu' / 'mps'
export DIFFBRUSH_STEPS=20      # DDIM sampling steps
```
