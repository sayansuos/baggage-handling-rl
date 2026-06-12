import time
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from configs.config import Experiment
from rl.sac.sac import run_sac
from simulator.environment.environment import Environment
from utils.logging import log_metrics, log_metrics_debug, log_training_metrics
from utils.plotting import (
    plot_animation,
    plot_grid,
    plot_rewards,
    plot_training_metrics,
)


def run_worker(exp: Experiment, worker_id: int, save=True) -> tuple[list[dict], int]:
    """
    One process handles multiple episodes on a single environment.
    """

    env = Environment(
        exp.env_config, exp.agent_config, exp.reward_config, exp.name, worker_id
    )

    histories = []
    steps = 0
    for i in range(exp.n_episodes):
        env.reset(seed=1234 + worker_id + i + 1)

        if save and worker_id == 0:
            plot_grid(env.grid_map.grid, "figures/simulation", exp.name)

        if i == exp.n_episodes - 1 and worker_id == 0:
            env.debug = True

        while not env.done():
            _, _, _, _, info = env.step()

        histories.append(info)
        steps += env.step_count

    return histories, steps


def run_simulation(experiments: list[Experiment]):
    """
    Run multiple episodes in parallel.
    """

    print(
        f"\nINFO: Starting SIMULATION with {len(experiments)} experiments\n", end="\r"
    )

    start = time.perf_counter()

    histories = []
    total_eps = 0
    total_steps = 0

    for k, exp in enumerate(experiments, start=1):

        print("=" * 60)
        print(f"INFO: Experiment {k}/{len(experiments)}")
        print(f"INFO: Name        : {exp.name}")
        print(f"INFO: Environments: {exp.n_envs}")
        print(f"INFO: Episodes    : {exp.n_episodes}")
        print("=" * 60)

        tasks = []
        worker_id = 0
        total_eps += exp.n_envs * exp.n_episodes

        for _ in range(exp.n_envs):
            tasks.append((exp, worker_id))
            worker_id += 1
        n_processes = min(cpu_count(), len(tasks))

        with Pool(processes=n_processes) as pool:
            results = list(
                tqdm(
                    pool.starmap(run_worker, tasks),
                    total=len(tasks),
                    desc=f"Simulation -- {exp.name}",
                )
            )

        for h, s in results:
            histories.extend(h)
            total_steps += s

    end = time.perf_counter()

    print(
        f"INFO: Resulting in {total_eps:,} episodes and {total_steps:,} steps. Execution time: {end - start:.6f} s.\n"
    )

    _ = log_metrics(histories, "logs/simulation")
    df_debug = log_metrics_debug(histories, "logs/simulation")
    plot_rewards(df_debug, "figures/simulation")

    print("\nINFO: TASK COMPLETED!\n")


def run_test(experiments: list[Experiment], render: bool = False):
    """
    Run some episodes and plot at each step.
    """

    print(f"\nINFO: Starting TEST with {len(experiments)} experiments...\n", end="\r")

    if render:
        plt.ion()
        _, ax = plt.subplots(figsize=(10, 8))

    histories = []

    for exp in experiments:

        env = Environment(
            env_config=exp.env_config,
            agent_config=exp.agent_config,
            reward_config=exp.reward_config,
            name=exp.name,
            env_id=1,
            debug=True,
        )
        env.reset(seed=1234)

        if render:
            env.render(ax=ax)
            plt.pause(0.001)

        while not env.done():
            _, _, _, _, info = env.step()

            if render:
                env.render(ax=ax)
                plt.pause(0.001)

        histories.append(info)

        if render:
            plt.ioff()

    _ = log_metrics(histories, "logs/test")
    df_debug = log_metrics_debug(histories, "logs/test")
    plot_rewards(df_debug, "figures/test")

    print("\nINFO: TASK COMPLETED!\n")


def run_save(
    experiments: list[Experiment],
    fps: int = 20,
):
    """
    Run one episode per experiment and save the animation.
    """

    print(f"\nINFO: Starting SAVE with {len(experiments)} experiments...\n", end="\r")

    frames = []

    for exp in experiments:

        env = Environment(exp.env_config, exp.agent_config, exp.reward_config, exp.name)
        env.reset(seed=1234)

        fig, ax = plt.subplots(figsize=(10, 8))
        env.render(ax)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.renderer.buffer_rgba()).copy()
        frames.append(frame)

        while not env.done():
            env.step()
            env.render(ax)
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.renderer.buffer_rgba()).copy()
            frames.append(frame)

    plt.close(fig)

    plot_animation(frames, "figures/test", "", fps)

    print("\nINFO: TASK COMPLETED!\n")


def run_train(
    experiments: list[Experiment],
    n_episodes: int | None = None,
):
    """
    Train a SAC agent on each experiment.
    """

    print(f"\nINFO: Starting TRAINING with {len(experiments)} experiments\n", end="\r")

    start = time.perf_counter()

    total_eps = 0
    total_steps = 0

    for exp in experiments:

        print("\n" + "=" * 60)
        print(f"TRAINING {exp.name}")
        print("=" * 60)

        history, _, metrics, best_score, _, steps = run_sac(
            exp=exp,
            worker_id=0,
            n_episodes=n_episodes or exp.n_episodes,
        )

        total_eps += n_episodes or exp.n_episodes
        total_steps += steps

        print(f"INFO: {exp.name} finished " f"(best score = {best_score:.3f})")

        df = log_training_metrics(metrics, "logs/train", exp.name)
        plot_training_metrics(df, "figures/train", exp.name)
        _ = log_metrics(history, "logs/train", exp.name)
        df_debug = log_metrics_debug(history, "logs/train", exp.name)

        plot_rewards(df_debug, "figures/train")

    end = time.perf_counter()

    print(
        f"INFO: Resulting in {total_eps:,} episodes and {total_steps:,} steps. Execution time: {end - start:.6f} s.\n"
    )
    print("\nINFO: TASK COMPLETED!\n")
