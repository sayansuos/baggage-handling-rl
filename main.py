import time
import imageio
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from simulator.configs.config import EnvConfig, AgentConfig, RewardConfig
from simulator.environment.environment import Environment
from simulator.utils.save_figures import save_grid
from simulator.utils.save_logs import save_as_df
from simulator.utils.load_config import (
    load_env_config,
    load_agent_config,
    load_reward_config,
)


def run_worker(args, save=True):
    """
    One process handles multiple episodes on a single environment.
    """

    env_config, agent_config, reward_config, n_episodes, worker_id = args
    np.random.seed(1234 + worker_id)
    env = Environment(env_config, agent_config, reward_config, worker_id)

    if save:
        save_grid(
            env.grid_map._grid,
            f"grid_{worker_id}.png",
            path="logs/",
            scale=10,
            show_grid=True,
        )

    histories = []
    for i in range(n_episodes):
        if i == n_episodes - 1:
            env._set_debug(True)
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step()
            done = all(terminated[a] or truncated[a] for a in terminated.keys())
        histories.append(info)
        env.reset(seed=1234 + worker_id + i + 1)

    return histories


def run_simulation(
    env_config: EnvConfig,
    agent_config: AgentConfig,
    reward_config: RewardConfig,
    nb_episode: int = 1000,
):
    """
    Run multiple episodes in parallel.
    """
    n_workers = cpu_count()
    episodes_per_worker = nb_episode // n_workers
    remainder = nb_episode % n_workers

    print(
        f"INFO: {n_workers} workers -- {episodes_per_worker} episodes per worker (+ {remainder})"
    )

    start = time.perf_counter()
    tasks = [
        (
            env_config,
            agent_config,
            reward_config,
            episodes_per_worker + (1 if i < remainder else 0),
            i + 1,
        )
        for i in range(n_workers)
    ]
    with Pool(processes=cpu_count()) as pool:
        results = list(
            tqdm(
                pool.imap(run_worker, tasks),
                total=len(tasks),
                desc="Simulation",
            )
        )
    histories = []
    for history in results:
        histories.extend(history)
    end = time.perf_counter()

    print(f"Execution time : {end - start:.6f} s")
    save_as_df(histories, "logs")


def run_test(
    env_config: EnvConfig,
    agent_config: AgentConfig,
    reward_config: RewardConfig,
    nb_episode: int = 3,
):
    """
    Run some episodes and plot at each step.
    """

    np.random.seed(1234)
    env = Environment(env_config, agent_config, reward_config)

    plt.ion
    for i in range(nb_episode):
        env.render()
        env.ax.set_title(
            f"Simulation test -- Episode {i+1}/{nb_episode} -- Step {env.step_count}"
        )
        plt.pause(0.01)
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step()
            env.render()
            env.ax.set_title(
                f"Simulation test -- Episode {i+1}/{nb_episode} -- Step {env.step_count}"
            )
            plt.pause(0.01)
            done = all(terminated[a] or truncated[a] for a in terminated.keys())
        env.reset()
    plt.ioff


def run_save(
    env_config: EnvConfig,
    agent_config: AgentConfig,
    reward_config: RewardConfig,
    file_name: str = "anim.mp4",
    path: str = "figures/",
    done: bool = False,
    fps: int = 30,
):
    """
    Run one episode and save the corresponding figures.
    """

    np.random.seed(1234)
    env = Environment(env_config, agent_config, reward_config)

    save_grid(
        env.grid_map.current_grid,
        f"grid.png",
        scale=10,
        show_grid=True,
    )
    save_grid(
        env._get_local_grid(env.agents[0]),
        f"grid_local.png",
        scale=50,
        show_grid=True,
    )

    writer = imageio.get_writer(path + file_name, fps=fps)
    env.render()
    env.ax.set_title(f"Simulation test -- Step {env.step_count}")
    plt.pause(0.01)
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step()
        done = all(terminated[a] or truncated[a] for a in terminated.keys())
        env.render()
        env.ax.set_title(f"Simulation test  -- Step {env.step_count}")
        env.fig.canvas.draw()
        frame = np.asarray(env.fig.canvas.renderer.buffer_rgba())
        writer.append_data(frame)
    writer.close()


if __name__ == "__main__":

    MODE = "test"

    env_config = load_env_config("simulator/configs/env_config/crossing_hard.yaml")
    agent_config = load_agent_config()
    reward_config = load_reward_config()

    if MODE == "simulation":
        run_simulation(env_config, agent_config, reward_config, 1000)
    elif MODE == "test":
        run_test(
            env_config,
            agent_config,
            reward_config,
        )
    elif MODE == "save":
        run_save(
            env_config,
            agent_config,
            reward_config,
        )
