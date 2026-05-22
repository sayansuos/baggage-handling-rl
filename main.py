import json
import time
import imageio
import numpy as np
import matplotlib.pyplot as plt
from simulator.utils.config import EnvConfig, AgentConfig, RewardConfig
from simulator.environment.environment import Environment
from simulator.utils.save_figures import save_grid


def run_simulation(env: Environment, nb_episode: int = 1000, save_occurence: int = 100):
    """
    Run multiple episodes.
    """

    start = time.perf_counter()
    while env.episode <= nb_episode:
        history = []
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step()
            history.append(info)
            done = all(terminated[a] or truncated[a] for a in terminated.keys())
        env.reset()
        if env.episode % save_occurence == 0:
            print(f"Episode {env.episode}/{nb_episode}")
            with open(f"logs/episode_{env.episode}.json", "w") as f:
                json.dump(history, f, indent=2)
            save_grid(
                env.grid_map.current_grid,
                f"grid_{env.episode}.png",
                scale=10,
                show_grid=True,
                path="logs/",
            )
    end = time.perf_counter()
    print(f"Execution time : {end - start:.6f} s")


def run_test(env: Environment):
    """
    Run one episode and plot at each step.
    """

    done = False
    plt.ion
    while not done:
        _, _, terminated, truncated, _ = env.step()
        env.render()
        plt.pause(0.05)
        done = all(terminated[a] or truncated[a] for a in terminated.keys())
    plt.ioff


def run_save(
    env: Environment,
    file_name: str = "anim.mp4",
    path: str = "figures/",
    done: bool = False,
    fps: int = 30,
):
    """
    Run one episode and save the corresponding figures.
    """

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
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step()
        done = all(terminated[a] or truncated[a] for a in terminated.keys())
        env.render()
        env.fig.canvas.draw()
        frame = np.asarray(env.fig.canvas.renderer.buffer_rgba())
        writer.append_data(frame)
    writer.close()


if __name__ == "__main__":

    MODE = "simulation"

    np.random.seed(123)

    env_config = EnvConfig()
    agent_config = AgentConfig()
    reward_config = RewardConfig()
    env = Environment(env_config, agent_config, reward_config)

    if MODE == "simulation":
        run_simulation(env, 1000)
    elif MODE == "test":
        run_test(env)
    elif MODE == "save":
        run_save(env)
