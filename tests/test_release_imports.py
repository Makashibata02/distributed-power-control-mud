"""Import smoke tests for the public Part-based package layout."""

from dpc_mud.parts.part1_two_user import experiments as part1_experiments
from dpc_mud.parts.part2_grouped_access import model as part2_model
from dpc_mud.parts.part2_grouped_access import receivers as part2_receivers
from dpc_mud.parts.part3_stochastic_optimization import monte_carlo, optimizer
from dpc_mud.workflows import (
    part2_baselines,
    part3_method_comparison,
    part3_monte_carlo_baselines,
    part3_pso_optimization,
    part3_sensitivity,
)


def main() -> None:
    modules = [
        part1_experiments,
        part2_model,
        part2_receivers,
        monte_carlo,
        optimizer,
        part2_baselines,
        part3_monte_carlo_baselines,
        part3_pso_optimization,
        part3_method_comparison,
        part3_sensitivity,
    ]
    assert all(module.__name__ for module in modules)
    print("test_release_imports: OK")


if __name__ == "__main__":
    main()
