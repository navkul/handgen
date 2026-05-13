# Model Weights

Do not commit DiffBrush weights into this product repo.

For this staging MVP, the default model path points at the existing research checkout:

```text
../04_cpu_neural_scout/vendor/DiffBrush/model_zoo/DiffBrush-ckpt.pt
```

When this folder becomes a standalone repo, keep weights outside Git and provide the path through:

```bash
export DIFFBRUSH_ROOT=/path/to/DiffBrush
export DIFFBRUSH_CHECKPOINT=/path/to/DiffBrush-ckpt.pt
export DIFFBRUSH_RUNNER=/path/to/run_diffbrush_single.py
export DIFFBRUSH_PYTHON=/path/to/python
```
