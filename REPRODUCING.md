# Reproducing the Results / 结果复现

## Environment

- Python 3.12
- NumPy 1.26.4
- Pandas 2.2.2
- Matplotlib 3.9.2
- PyYAML 6.0.1

Install the package in editable mode from the repository root:

```bash
python -m pip install -e .
```

## Unified workflow

The command below runs Part I, then Part II, and finally the four Part III stages in dependency order:

```bash
python -m dpc_mud reproduce --quick --output-dir results/generated
```

Remove `--quick` for the full search and Monte Carlo settings used by the research snapshot. Both modes use the fixed random seeds in `configs/part2_grouped_access.yaml` and `configs/part3_stochastic_optimization.yaml`.

The workflow writes:

1. `part1_two_user`: analytical tables and figures.
2. `part2_grouped_access`: formula-based baselines and grouped local search.
3. `part3_stochastic_optimization`: Monte Carlo baseline evaluation, PSO, comparison, and sensitivity scans.
4. `run_manifest.json`: mode, software versions, configurations, and repository-relative output paths.

Every run creates timestamped experiment subdirectories. Generated files go under `results/generated` and do not overwrite `results/published`.

## Published snapshot

`results/published` contains the reviewed, compact snapshot. Large per-sample Monte Carlo tables and the complete Part II search trace are intentionally omitted; the code and fixed seeds can regenerate them. A compact representative trace is retained for inspection.

## 中文备注

快速模式用于安装验证和持续集成，不代表终稿的完整计算预算。完整模式使用配置文件中的正式搜索规模。各实验输出均使用仓库相对路径，不记录用户名、主机名或本机绝对路径。
