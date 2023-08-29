from __future__ import annotations

from julia import Julia

jl = Julia(threads=8)

import pathlib
from typing import TYPE_CHECKING

import numpy as np
from eta_utility import LOG_DEBUG, get_logger
from eta_utility.eta_x import ConfigOptRun, ETAx, LinearSchedule
from eta_utility.eta_x.agents.nsga2 import Nsga2
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

if TYPE_CHECKING:
    from typing import Any

    from stable_baselines3.common.base_class import BaseAlgorithm


get_logger(level=LOG_DEBUG, format="logname")


def main() -> None:
    general_config: dict[str, Any] = {
        "root_path": pathlib.Path(__file__).parent,
        "config_name": "production",
        "relpath_config": ".",
    }

    config_overwrite: dict[str, dict[str, Any]] = {
        "environment_specific": {
            "orders_file": "orders2.json",
        },
        "agent_specific": {
            "population": 100,
            "crossovers": 0.3,
            "mutations": 0.02,
            "learning_rate": LinearSchedule(0.5, 1.0),
        },
    }

    # Perform schedule optimizations and plot their results.
    series_name = "tests2"
    etax = ETAx(**general_config, config_overwrite=config_overwrite)
    learn(etax, series_name, "test_2prod_noinit")  # , expert_init="spt")

    config_overwrite["environment_specific"]["orders_file"] = "orders_15prod.json"
    config_overwrite["agent_specific"]["n_generations"] = 1000
    etax = ETAx(**general_config, config_overwrite=config_overwrite)
    learn(etax, series_name, "15prod_spt_3", expert_init="spt")
    plot(etax, series_name, "15prod_spt_3", solutions=[142, 155], debug_annotations=False)

    # Create "expert" schedules using the SPT dispatching rule and plot them.
    series_name = "expertmodels"
    config_overwrite["environment_specific"].update(
        {
            "orders_file": "orders_5prod.json",
            "varsecondmap": [0],
        }
    )
    config_overwrite["agent_specific"].update({"population": 5, "n_generations": 500})

    etax = ETAx(**general_config, config_overwrite=config_overwrite)
    expert_schedules_plot(etax, series_name, "5prod_spt_init", "spt")


def learn(etax: ETAx, series_name: str, run_name: str, expert_init: str | None = None) -> None:
    """Execute the algorithm training.

    :param etax: The etax class, which manages the optimization.
    :param series_name: A name for the series of optimization runs.
    :param run_name: A name for each optimization run.
    :param expert_init: This parameter can contain the name of an expert initialization rule, like SPT.
    """
    etax.learn(series_name, run_name, callbacks=ProductionCallback(20, expert_init))


def plot(etax: ETAx, series_name: str, run_name: str, *, solutions: list[int] | str, debug_annotations=True) -> None:
    """Plot some optimization results,

    :param etax: The etax class, which manges the optimization.
    :param series_name: A name for the series of optimization runs.
    :param run_name: A name for each optimization run.
    :param solutions: List of solutions to plot.
    :param debug_annotation: By default some additional annotation help with debugging. Set this to false to have more 
                             beautiful plots.
    """
    with etax.prepare_environments_models(series_name, run_name):
        assert isinstance(etax.environments, VecEnv)
        assert isinstance(etax.config_run, ConfigOptRun)

        set_environment_buffers(etax.environments, etax.model)
        if solutions == "all":
            etax.environments.env_method(
                "render",
                solutions,
                path=etax.config_run.path_series_results.as_posix(),
                filename=f"{run_name}_solplots",
                fileextension="svg",
                debug_annotations=debug_annotations,
                indices=1,
            )
        else:
            for solution in solutions:
                etax.environments.env_method(
                    "render",
                    solution,
                    path=etax.config_run.path_series_results.as_posix(),
                    filename=f"{run_name}_solplots",
                    fileextension="svg",
                    debug_annotations=debug_annotations,
                    indices=1,
                )


class ProductionCallback(BaseCallback):
    """
    Callback to tell the environment about the entire current solution space.
    Adds an on_step callback used by the environment for plotting functions.

    :param save_freq: Frequency how often the solution space should be transferred
                      to the environment and plotted.
    :param expert_init: Name of a dispatching rule to use for initialization.
    :param kwargs: Additional arguments for the BaseCallback superclass."""

    def __init__(self, save_freq: int, expert_init: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.save_freq = save_freq
        self.n_calls = 0
        self.expert_init = expert_init

    def _on_training_start(self) -> None:
        """Set up the parent generation with solutions generated by an expert system."""
        assert isinstance(self.model, Nsga2)
        assert isinstance(self.training_env, VecEnv)
        if self.expert_init is None:
            return

        expert_actions = self.training_env.env_method(
            "generate_expert_schedules", self.expert_init, self.model.population, indices=1
        )

        self.model.generation_parent = self.model._jl_setup_generation(
            expert_actions[0]["events"], expert_actions[0]["variables"], self.model._max_value
        )
        self.model._evaluate(self.model.generation_parent)

    def _on_step(self) -> bool:
        """Set environment buffers and call the render functions when save_freq steps have passed."""
        if self.n_calls % self.save_freq == 0:
            assert isinstance(self.training_env, VecEnv)

            set_environment_buffers(self.training_env, self.model)
            self.training_env.env_method("render", mode="human", indices=1)

        return True


def set_environment_buffers(env: Any | VecEnv | None, model: BaseAlgorithm | None) -> None:
    """Set the actions and rewards buffers of an environment to enable it to plot specific solutions.

    :param env: The (vectorized) environment object-
    :param model: The model which provides the results.
    """
    assert isinstance(env, VecEnv)
    assert isinstance(model, Nsga2)
    env.set_attr("buffer_events", model.last_evaluation_actions["events"])
    env.set_attr("buffer_variables", model.last_evaluation_actions["variables"])
    env.set_attr("buffer_rewards", model.last_evaluation_rewards)
    env.set_attr("buffer_fronts", model.last_evaluation_fronts)


def expert_schedules_plot(
    etax: ETAx, series_name: str, run_name: str, expert_init: str, plot: str | int = "all"
) -> None:
    """Create an plot some expert schedules.

    :param etax: The etax class which manages the optimization.
    :param series_name: A name for the series of optimization runs.
    :param run_name: A name for each optimization run.
    :param expert_init: Name of a dispatching rule to use for initialization.
    :param plot: List of solutions to plot.
    """
    with etax.prepare_environments_models(series_name, run_name):
        assert isinstance(etax.environments, VecEnv)
        assert isinstance(etax.model, Nsga2)
        assert isinstance(etax.config_run, ConfigOptRun)

        expert_actions = etax.environments.env_method(
            "generate_expert_schedules", expert_init, etax.model.population, indices=1
        )
        solutions = etax.model._jl_setup_generation(
            expert_actions[0]["events"], expert_actions[0]["variables"], etax.model._max_value
        )

        solutions, retries = etax.model._evaluate(solutions)

        etax.environments.set_attr("buffer_rewards", np.vstack([sol.reward for sol in solutions]))
        etax.environments.set_attr("buffer_events", expert_actions[0]["events"])
        etax.environments.set_attr("buffer_variables", expert_actions[0]["variables"])
        etax.environments.set_attr(
            "buffer_fronts",
            [list(range(etax.model.population - 1)), list(range(etax.model.population - 1, etax.model.population))],
        )

        etax.environments.env_method(
            "render",
            plot,
            path=etax.config_run.path_series_results.as_posix(),
            filename=f"{run_name}_expertplots",
            fileextension="svg",
            indices=1,
        )


if __name__ == "__main__":
    main()
