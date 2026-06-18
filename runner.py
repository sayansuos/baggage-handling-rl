import time
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from configs.config import Experiment
from rl.sac.sac import evaluate_sac, run_sac
from simulator.environment.environment import Environment
from utils.logging import log_debug, log_rewards, log_train
from utils.plotting import (
    plot_animation,
    plot_grid,
    plot_performances,
    plot_rewards,
    plot_velocities,
)


def run_worker(exp: Experiment, worker_id: int) -> tuple[list[dict], list[dict], int]:
    """
    One process handles multiple episodes on a single environment.
    """

    env = Environment(
        exp.env_config, exp.agent_config, exp.reward_config, exp.name, worker_id
    )

    history = []
    debug = []
    steps = 0

    for i in range(exp.n_episodes):
        env.reset(seed=1234 + worker_id + i + 1)

        env.debug = i == exp.n_episodes - 1 and worker_id == 0

        while not env.done():
            _, rewards, _, _, info = env.step()

            if env.debug:  # Store metrics for debug logs
                for agent_id, info in info.items():
                    debug.append(
                        {
                            "experiment": env.name,
                            "episode": env.episode,
                            "step": env.step_count,
                            "agent": agent_id,
                            "pos_x": info["pos_x"],
                            "pos_y": info["pos_y"],
                            "distance_to_goal": info["distance_to_goal"],
                            "heading_error": info["heading_error"],
                            "min_obstacle_distance": info["min_obstacle_distance"],
                            "v": info["v"],
                            "omega": info["omega"],
                            "reward": rewards[agent_id],
                            "reward_progress": info["reward_progress"],
                            "reward_rotation": info["reward_rotation"],
                            "reward_safety": info["reward_safety"],
                            "reward_collision": info["reward_collision"],
                            "state": info["state"],
                        }
                    )

        env.debug = False
        if not env.debug:
            history.append(info)

        steps += env.step_count

    return history, debug, steps


def run_simulation(experiments: list[Experiment]):
    """
    Run multiple episodes in parallel.
    """

    start = time.perf_counter()
    print(f"\n[ SIMULATION ] {len(experiments)} experiment(s)\n")

    histories = []
    debugs = []
    total_eps = 0
    total_steps = 0

    for exp in experiments:
        print(f"[ START ] {exp.name}")

        tasks = [(exp, worker_id) for worker_id in range(exp.n_envs)]
        total_eps += exp.n_envs * exp.n_episodes

        n_processes = min(cpu_count(), len(tasks))

        with Pool(processes=n_processes) as pool:
            results = list(
                tqdm(
                    pool.starmap(run_worker, tasks),
                    total=len(tasks),
                    desc=f"Simulation -- {exp.name}",
                )
            )

        for history, debug, steps in results:
            histories.extend(history)
            debugs.extend(debug)
            total_steps += steps

        log_train(histories, "logs/simulation", exp.name)
        log_debug(debugs, "logs/simulation", exp.name)
        log_rewards(histories, "logs/simulation", exp.name)

        print(f"[ DONE ] {exp.name}")

    duration = time.perf_counter() - start

    print(
        f"\n[ SUMMARY ] {total_eps:,} episode(s) | "
        f"{total_steps:,} step(s) | {duration:.1f}s\n"
    )


def run_test(experiments: list[Experiment], render: bool = False):
    """
    Run some episodes and plot at each step.
    """

    print(f"\n[ TEST ] {len(experiments)} experiment(s)\n")

    if render:
        plt.ion()
        _, ax = plt.subplots(figsize=(10, 8))

    histories = []
    debugs = []

    for exp in experiments:

        print(f"[ START ] {exp.name}")

        env = Environment(
            env_config=exp.env_config,
            agent_config=exp.agent_config,
            reward_config=exp.reward_config,
            name=exp.name,
            env_id=1,
            debug=True,
        )
        env.reset(seed=1234)
        plot_grid(env.grid_map.grid, "figures/test", exp.name)

        if render:
            env.render(ax=ax)
            plt.pause(0.001)

        while not env.done():
            _, rewards, _, _, info = env.step()

            for agent_id, info in info.items():
                debugs.append(
                    {
                        "experiment": env.name,
                        "episode": env.episode,
                        "step": env.step_count,
                        "agent": agent_id,
                        "pos_x": info["pos_x"],
                        "pos_y": info["pos_y"],
                        "distance_to_goal": info["distance_to_goal"],
                        "heading_error": info["heading_error"],
                        "min_obstacle_distance": info["min_obstacle_distance"],
                        "v": info["v"],
                        "omega": info["omega"],
                        "reward": rewards[agent_id],
                        "reward_progress": info["reward_progress"],
                        "reward_rotation": info["reward_rotation"],
                        "reward_safety": info["reward_safety"],
                        "reward_collision": info["reward_collision"],
                        "state": info["state"],
                    }
                )

            if render:
                env.render(ax=ax)
                plt.pause(0.001)

        histories.append(info)
        print(f"[ DONE ] {exp.name}")

        if render:
            plt.ioff()

    log_train(histories, "logs/test", "test")
    log_debug(debugs, "logs/test", "test")
    log_rewards(histories, "logs/test", "test")

    print("\n[ TEST ] completed\n")


def run_save(
    experiments: list[Experiment],
    fps: int = 20,
):
    """
    Run one episode per experiment and save the animation.
    """

    print(f"\n[ SAVE ] {len(experiments)} experiment(s)\n")

    frames = []

    for exp in experiments:
        print(f"[ START ] {exp.name}")

        env = Environment(exp.env_config, exp.agent_config, exp.reward_config, exp.name)
        env.reset(seed=1234)

        fig, ax = plt.subplots(figsize=(10, 8))

        env.render(ax)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.renderer.buffer_rgba()).copy())

        while not env.done():
            env.step()
            env.render(ax)
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.renderer.buffer_rgba()).copy()
            frames.append(frame)

        print(f"[ DONE ] {exp.name}")

    plt.close(fig)

    plot_animation(frames, "figures/test", "", fps)

    print("\n[ SAVE ] completed\n")


def run_train(experiments: list[Experiment], n_steps: list[int] = [5000]):
    """
    Train a SAC agent on each experiment.
    """

    np.random.seed(1234)

    start = time.perf_counter()
    print(f"\n[ TRAIN ] {len(experiments)} experiment(s)\n")

    agent = None

    for i, exp in enumerate(experiments):
        history, debug, agent = run_sac(exp=exp, n_steps=n_steps[i], agent=agent)

        df_train = log_train(history, "logs/train", exp.name)
        log_debug(debug, "logs/train", exp.name)
        df_rewards = log_rewards(history, "logs/train", exp.name)

        plot_performances(df_train, "figures/train", exp.name)
        plot_velocities(df_train, "figures/train", exp.name)
        plot_rewards(df_rewards, "figures/train", exp.name)

    duration = time.perf_counter() - start

    print(
        f"\n[ SUMMARY ] "
        f"{len(experiments)} experiment(s) completed "
        f"in {duration:.1f}s\n"
    )


def run_evaluation(
    experiments: list[Experiment],
    n_episodes: int = 100,
    n_render: int = 5,
):

    np.random.seed(1234)

    for exp in experiments:

        history, frames = evaluate_sac(
            exp=exp,
            n_episodes=n_episodes,
            n_render=n_render,
            trained_agent_id="agent_1",
        )

        df_train = log_train(history, "logs/eval", exp.name)
        df_rewards = log_rewards(history, "logs/eval", exp.name)

        plot_performances(df_train, "figures/eval", exp.name)
        plot_velocities(df_train, "figures/eval", exp.name)
        plot_rewards(df_rewards, "figures/eval", exp.name)

        if n_render > 0:
            plot_animation(frames, "figures/eval", exp.name, 10)
