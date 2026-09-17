# Reproducibility workflows

The public workflow is intentionally exposed through one command:

```bash
python -m dpc_mud reproduce --quick --output-dir results/generated
```

Its implementations live under `src/dpc_mud/workflows`:

- `part2_baselines.py`
- `part3_monte_carlo_baselines.py`
- `part3_pso_optimization.py`
- `part3_method_comparison.py`
- `part3_sensitivity.py`

The unified command passes each generated result directory to the next dependent experiment, preventing manual selection of timestamped folders.
