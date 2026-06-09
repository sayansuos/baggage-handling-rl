import time
import imageio
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from simulator.configs.config import Experiment
from simulator.environment.environment import Environment
from simulator.utils.save_figures import save_grid, save_rewards
from simulator.utils.save_logs import save_as_df
from rl.SAC.sac import run_sac
from rl.utils import save_training_metrics


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
            f"grid_{exp.name}",
            path="logs/",
            scale=10,
            show_grid=True,
        )

    histories = []
    steps = 0
    for i in range(exp.n_episodes):
        if i == exp.n_episodes - 1 and worker_id == 0:
            env._set_debug(True)
        while not env.done:
            _, _, _, _, info = env.step()
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
    print(f"\nINFO: Starting SIMULATION with {len(experiments)} experiments\n")
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

    print(f"\nINFO: Ending SIMULATION with {len(experiments)} experiments:")
    print(
        f"INFO: Resulting in {"{:,}".format(total_eps)} episodes and {"{:,}".format(total_steps)} steps.\n"
    )
    print(f"INFO: Execution time: {end - start:.6f} s")
    print("\nINFO: Saving files...\n")
    _, df_debug = save_as_df(histories, "simulation", "logs")
    save_rewards(df_debug, "reward_tracking", "logs")
    print("\nINFO: TASK COMPLETED!\n")


def run_test(experiments: list[Experiment], render: bool = False):
    """
    Run some episodes and plot at each step.
    """

    np.random.seed(1234)
    plt.figure(figsize=(10, 8))
    plt.ion()
    print(f"\nINFO: Starting TEST with {len(experiments)} experiments...\n")
    histories = []
    for exp in experiments:
        print(f"\nINFO: Testing {exp.name}...\n")
        env = Environment(
            exp.env_config, exp.agent_config, exp.reward_config, exp.name, 1, True
        )
        if render:
            env.render()
            env.ax.set_title(f"TEST {env.name} -- STEP {env.step_count}")
            plt.pause(0.001)
        while not env.done:
            _, _, _, _, info = env.step()
            if render:
                env.render()
                env.ax.set_title(f"TEST {env.name} -- STEP {env.step_count}")
                plt.pause(0.001)
        histories.append(info)
        env.reset()
    plt.ioff()
    print(f"\nINFO: Ending TEST with {len(experiments)} experiments.\n")
    print("\nINFO: Saving files...\n")
    _, df_debug = save_as_df(histories, "test", "figures")
    save_rewards(df_debug, "reward_tracking", "figures")
    print("\nINFO: TASK COMPLETED!\n")


def run_save(
    experiments: list[Experiment],
    file_name: str = "anim.mp4",
    path: str = "figures/",
    fps: int = 20,
):
    """
    Run one episode and save the corresponding figures.
    """

    np.random.seed(1234)
    print(f"\nINFO: Starting SAVE with {len(experiments)} experiments...\n")
    writer = imageio.get_writer(path + file_name, fps=fps)
    for exp in experiments:
        print(f"\nINFO: Saving {exp.name}...\n")
        env = Environment(exp.env_config, exp.agent_config, exp.reward_config, exp.name)
        save_grid(
            env.grid_map.grid,
            f"grid_{exp.name}",
            scale=10,
            show_grid=True,
        )
        env.render()
        env.ax.set_title(f"TEST {env.name} -- STEP {env.step_count}")
        while not env.done:
            _, _, _, _, _ = env.step()
            env.render()
            env.ax.set_title(f"TEST {env.name} -- STEP {env.step_count}")
            env.fig.canvas.draw()
            frame = np.asarray(env.fig.canvas.renderer.buffer_rgba())
            writer.append_data(frame)
    writer.close()
    print(f"\nINFO: Ending SAVE with {len(experiments)} experiments.\n")


def run_train(
    experiments: list[Experiment],
    n_episodes: int | None = None,
):
    """
    Train a SAC agent on each experiment.
    """
    start = time.perf_counter()
    print(f"\nINFO: Starting TRAINING with {len(experiments)} experiments\n")

    for exp in experiments:
        print("\n" + "=" * 60)
        print(f"TRAINING {exp.name}")
        print("=" * 60)

        history, metrics, best_score, _ = run_sac(
            exp=exp,
            worker_id=1,
            n_episodes=n_episodes or exp.n_episodes,
        )

        print("INFO: Saving files...")
        save_training_metrics(
            metrics,
            f"{exp.name}_training",
            path="logs/training",
        )

        print(f"INFO: {exp.name} finished " f"(best score = {best_score:.3f})")

    end = time.perf_counter()
    print(f"\nINFO: Ending SIMULATION with {len(experiments)} experiments:")
    print(f"INFO: Execution time: {end - start:.6f} s")
    print("\nINFO: TASK COMPLETED!\n")
