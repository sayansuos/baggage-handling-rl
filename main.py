import time
import imageio
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from itertools import chain

from simulator.configs.config import EnvConfig, AgentConfig, RewardConfig, Experiment
from simulator.environment.environment import Environment
from simulator.utils.save_figures import save_grid
from simulator.utils.save_logs import save_as_df
from simulator.utils.load_config import (
    load_env_config,
    load_agent_config,
    load_reward_config,
    load_experiments,
)


def run_worker(exp: Experiment, worker_id: int, save=True) -> tuple[list[dict], int]:
    """
    One process handles multiple episodes on a single environment.
    """

    env = Environment(
        exp.env_config, exp.agent_config, exp.reward_config, exp.name, worker_id
    )

    if save and worker_id == 0:
        save_grid(
            env.grid_map.grid,
            f"grid_{exp.name}.png",
            path="logs/",
            scale=10,
            show_grid=True,
        )

    histories = []
    steps = 0
    for i in range(exp.n_episodes):
        if i == exp.n_episodes - 1 and worker_id == 0:
            env._set_debug(True)
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step()
            done = all(terminated[a] or truncated[a] for a in terminated.keys())
        histories.append(info)
        steps += env.total_step
        env.reset(seed=1234 + worker_id + i + 1)

    return histories, steps


def run_simulation(experiments: list[Experiment]):
    """
    Run multiple episodes in parallel.
    """

    start = time.perf_counter()
    histories = []
    total_eps = 0
    total_steps = 0
    print(f"\nINFO: Starting simulation with {len(experiments)} experiments\n")
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

    print(f"\nINFO: Starting simulation with {len(experiments)} experiments:")
    print(
        f"INFO: Resulting in {"{:,}".format(total_eps)} episodes and {"{:,}".format(total_steps)} steps.\n"
    )
    print(f"Execution time : {end - start:.6f} s")
    save_as_df(histories, "logs")


def run_test(
    experiments: list[Experiment],
    nb_episode: int = 3,
):
    """
    Run some episodes and plot at each step.
    """

    np.random.seed(1234)

    plt.figure(figsize=(10, 8))
    plt.ion()
    for exp in experiments:
        env = Environment(exp.env_config, exp.agent_config, exp.reward_config, exp.name)
        for i in range(nb_episode):
            env.render()
            env.ax.set_title(
                f"SIMULATION TEST {env.name} -- Episode {i+1}/{nb_episode} -- Step {env.step_count}"
            )
            plt.pause(0.01)
            done = False
            while not done:
                _, _, terminated, truncated, _ = env.step()
                env.render()
                env.ax.set_title(
                    f"SIMULATION TEST {env.name} -- Episode {i+1}/{nb_episode} -- Step {env.step_count}"
                )
                plt.pause(0.01)
                done = all(terminated[a] or truncated[a] for a in terminated.keys())
            env.reset()
    plt.ioff()


def run_save(
    experiments: list[Experiment],
    file_name: str = "anim.mp4",
    path: str = "figures/",
    done: bool = False,
    fps: int = 20,
):
    """
    Run one episode and save the corresponding figures.
    """

    np.random.seed(1234)
    writer = imageio.get_writer(path + file_name, fps=fps)
    for exp in experiments:
        env = Environment(exp.env_config, exp.agent_config, exp.reward_config, exp.name)
        save_grid(
            env.grid_map.current_grid,
            f"grid_{exp.name}.png",
            scale=10,
            show_grid=True,
        )
        env.render()
        env.ax.set_title(f"SIMULATION TEST {env.name} -- Step {env.step_count}")
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step()
            done = all(terminated[a] or truncated[a] for a in terminated.keys())
            env.render()
            env.ax.set_title(f"SIMULATION TEST {env.name} -- Step {env.step_count}")
            env.fig.canvas.draw()
            frame = np.asarray(env.fig.canvas.renderer.buffer_rgba())
            writer.append_data(frame)
    writer.close()


if __name__ == "__main__":

    MODE = "save"

    env_config = load_env_config("simulator/configs/env_config/crossing_hard.yaml")
    agent_config = load_agent_config()
    reward_config = load_reward_config()
    experiments = load_experiments()

    if MODE == "simulation":
        run_simulation(experiments)
    elif MODE == "test":
        run_test(experiments)
    elif MODE == "save":
        run_save(experiments)
