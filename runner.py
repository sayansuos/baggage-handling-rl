import random
import time

import matplotlib.pyplot as plt
import numpy as np

from configs.config import Experiment
from rl.sac.sac import evaluate_sac, load_agent, run_sac
from simulator.environment.environment import Environment
from utils.logging import log_debug, log_rewards, log_train
from utils.plotting import plot_animation, plot_figures


def run_demo(
    experiments: list[Experiment],
    best_exp: Experiment,
    render: bool = False,
    trained_agent_id: str = "agent_1",
):
    """Run a trained policy on one episode for each experiment, with optional real-time rendering."""

    print(f"\n[ TEST ] {len(experiments)} experiment(s)\n")

    if render:
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig, ax = None, None

    for exp in experiments:
        print(f"[ START ] {exp.name}")

        env = Environment(
            env_config=exp.env_config,
            agent_config=exp.agent_config,
            reward_config=exp.reward_config,
            name=exp.name,
        )
        agent = load_agent(env, best_exp, trained_agent_id)
        agent.load_checkpoints()

        state, _ = env.reset(seed=1234)

        if render:
            env.render(ax=ax)
            plt.pause(0.001)

        while not env.dones[trained_agent_id]:
            actions = {}
            actions[trained_agent_id] = agent.choose_action(
                state[trained_agent_id],
                deterministic=True,
            )
            for amr in env.agents:
                if amr.id != trained_agent_id:
                    actions[amr.id] = None

            next_state, _, _, _, _ = env.step(actions)
            state = next_state

            if render:
                env.render(ax=ax)
                plt.pause(0.001)

        print(f"[ DONE ] {exp.name}")

    if render:
        plt.ioff()
        plt.close(fig)

    print("\n[ TEST ] completed\n")


def run_animation(
    experiments: list[Experiment],
    best_exp: Experiment,
    path: str,
    file_name: str,
    fps: int = 20,
    trained_agent_id: str = "agent_1",
):
    """Run one episode for each experiment and save the rendered trajectory as an animation."""

    print(f"\n[ SAVE ] {len(experiments)} experiment(s)\n")

    frames = []

    for exp in experiments:
        print(f"[ START ] {exp.name}")

        env = Environment(exp.env_config, exp.agent_config, exp.reward_config, exp.name)
        agent = load_agent(env, best_exp, trained_agent_id)
        agent.load_checkpoints()
        state, _ = env.reset(seed=1234)

        fig, ax = plt.subplots(figsize=(10, 8))

        env.render(ax)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.renderer.buffer_rgba()).copy())

        while not env.dones[trained_agent_id]:
            actions = {}
            actions[trained_agent_id] = agent.choose_action(
                state[trained_agent_id],
                deterministic=True,
            )
            for amr in env.agents:
                if amr.id != trained_agent_id:
                    actions[amr.id] = None

            next_state, _, _, _, _ = env.step(actions)
            state = next_state

            env.render(ax)
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.renderer.buffer_rgba()).copy()
            frames.append(frame)

        print(f"[ DONE ] {exp.name}")

    plt.close(fig)

    plot_animation(frames, path, file_name, fps)

    print("\n[ SAVE ] completed\n")


def run_train_all(
    experiments: list[Experiment],
    exp_name: str,
    previous_exp: Experiment | None,
    n_steps: int,
    chunk_steps: int,
):
    """Train a single policy by alternating between multiple scenarios and log the training results."""

    np.random.seed(1234)

    total_steps = 0
    histories = []
    debugs = []

    while total_steps < n_steps:

        env = random.choice(experiments)
        print(
            f"[ TRAIN MIXED ] "
            f"Step {total_steps:06d}/{n_steps:06d} | "
            f"Scenario = {env.name}"
        )

        steps = min(chunk_steps, n_steps - total_steps)
        exp = Experiment(
            name=exp_name,
            env_config=env.env_config,
            agent_config=env.agent_config,
            reward_config=env.reward_config,
            n_steps=steps,
        )

        history, debug, _ = run_sac(
            exp=exp, previous_exp=previous_exp, warmup_steps=0, reset_frequency=1
        )

        for row in history:
            row["scenario"] = env.name

        for row in debug:
            row["scenario"] = env.name

        histories.extend(history)
        debugs.extend(debug)

        previous_exp = exp
        total_steps += steps

    for i, history in enumerate(histories):
        history["episode"] = i

    for i, d in enumerate(debug):
        d["episode"] = i

    df_train = log_train(histories, "logs/train", exp_name)
    log_debug(debugs, "logs/train", exp_name)
    df_rewards = log_rewards(histories, "logs/train", exp_name)

    plot_figures(df_train, df_rewards, "figures/train", exp_name)


def run_train(
    experiments: list[Experiment],
    previous_exp: Experiment | None = None,
    n_steps: int = 200_000,
    chunk_steps: int = 2_000,
):
    """Train a policy for each experiment and save the corresponding logs and figures."""

    np.random.seed(1234)

    start = time.perf_counter()
    print(f"\n[ TRAIN ] {len(experiments)} experiment(s)\n")

    for exp in experiments:

        if "mixed_curriculum" not in exp.name:
            history, debug, _ = run_sac(exp=exp, previous_exp=previous_exp)

            df_train = log_train(history, "logs/train", exp.name)
            log_debug(debug, "logs/train", exp.name)
            df_rewards = log_rewards(history, "logs/train", exp.name)
            plot_figures(df_train, df_rewards, "figures/train", exp.name)

            previous_exp = exp

        else:
            run_train_all(experiments, exp.name, previous_exp, n_steps, chunk_steps)

    duration = time.perf_counter() - start

    print(
        f"\n[ SUMMARY ] "
        f"{len(experiments)} experiment(s) completed "
        f"in {duration:.1f}s\n"
    )


def run_validation(
    experiments: list[Experiment],
    n_episodes: int = 100,
    n_render: int = 5,
    agent=None,
):
    """Evaluate a trained policy on the validation scenarios and save the resulting metrics and animations."""

    np.random.seed(1234)

    start = time.perf_counter()
    print(f"\n[ VALIDATION ] {len(experiments)} experiment(s)\n")

    for exp in experiments:

        history, frames = evaluate_sac(
            exp=exp,
            agent=agent,
            n_episodes=n_episodes,
            n_render=n_render,
            trained_agent_id="agent_1",
        )

        log_train(history, "logs/validation", exp.name)
        log_rewards(history, "logs/validation", exp.name)

        if n_render > 0:
            plot_animation(frames, "figures/validation", exp.name, 10)

    duration = time.perf_counter() - start

    print(
        f"\n[ SUMMARY ] "
        f"{len(experiments)} experiment(s) completed "
        f"in {duration:.1f}s\n"
    )


def run_evaluation(
    experiments: list[Experiment],
    policy: Experiment,
    n_episodes: int = 100,
    n_render: int = 5,
):
    """Evaluate a trained policy on a set of scenarios and save the corresponding performance metrics and animations."""

    np.random.seed(1234)

    start = time.perf_counter()
    print(f"\n[ EVALUATION ] {len(experiments)} experiment(s)\n")

    for scenario in experiments:

        exp = Experiment(
            name=policy.name,
            env_config=scenario.env_config,
            agent_config=scenario.agent_config,
            reward_config=scenario.reward_config,
            n_steps=scenario.n_steps,
        )

        history, frames = evaluate_sac(
            exp=exp,
            n_episodes=n_episodes,
            n_render=n_render,
        )

        log_train(history, "logs/evaluation", scenario.name)
        log_rewards(history, "logs/evaluation", scenario.name)

        if n_render > 0:
            plot_animation(frames, "figures/evaluation", scenario.name, 10)

    duration = time.perf_counter() - start

    print(
        f"\n[ SUMMARY ] "
        f"{len(experiments)} experiment(s) completed "
        f"in {duration:.1f}s\n"
    )
