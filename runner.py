import random
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from configs.config import Experiment
from rl.sac.sac import evaluate_sac, load_agent, run_sac
from simulator.environment.environment import Environment
from utils.logging import log_debug, log_rewards, log_train
from utils.plotting import plot_animation, plot_figures


def run_demo(
    experiments: list[Experiment],
    policy: Experiment,
    trained_agent_id: str = "agent_1",
) -> None:
    """
    Run trained policies and interactively navigate through the rendered frames.
    """

    # Generate all the scenarios frames
    frames = run_animation(
        experiments=experiments,
        policy=policy,
        trained_agent_id=trained_agent_id,
    )

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.15)

    # Initialize frame
    current_frame = 0
    paused = True

    # Display the current frame
    def render():
        ax.clear()
        ax.imshow(frames[current_frame])
        ax.axis("off")

        # Refresh the figure without blocking the interface
        fig.canvas.draw_idle()

    # Advance automatically to the next frame
    def update(_):
        nonlocal current_frame, paused

        if not paused:
            if current_frame < len(frames) - 1:
                current_frame += 1
                render()
            else:
                paused = True
                animation.event_source.stop()

    # Handle keyboard navigation between frames
    def on_key(event):
        nonlocal current_frame, paused

        if event.key == "left" and current_frame > 0:
            current_frame -= 1
            render()

        elif event.key == "right" and current_frame < len(frames) - 1:
            current_frame += 1
            render()

        elif event.key == " ":
            paused = not paused

            if paused:
                animation.event_source.stop()
            else:
                animation.event_source.start()

    # Associate keyboard events with the navigation callback
    fig.canvas.mpl_connect("key_press_event", on_key)

    # Create the animation
    animation = FuncAnimation(
        fig,
        update,
        interval=100,
        cache_frame_data=False,
    )

    # Start paused
    animation.event_source.stop()

    # Display the first frame and start the interactive viewer
    render()
    plt.show()


def run_animation(
    experiments: list[Experiment],
    policy: Experiment,
    path: str | None = None,
    file_name: str | None = None,
    fps: int = 20,
    trained_agent_id: str = "agent_1",
) -> list[np.asarray]:
    """
    Run one episode for each experiment and return the rendered frames.
    If path and file_name are provided, also save the frames as an animation.
    """

    print(f"\n[ ANIMATION ] {len(experiments)} experiment(s)\n")

    # Initialize the frames
    frames = []

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Run one episode for each experiment
    for exp in experiments:
        print(f"[ START ] {exp.name}")

        # Create the environment from the experiment configuration
        env = Environment(
            env_config=exp.env_config,
            agent_config=exp.agent_config,
            reward_config=exp.reward_config,
            name=exp.name,
        )

        # Create the SAC agent using the selected trained policy
        agent = load_agent(env=env, exp=policy, trained_agent_id=trained_agent_id)
        agent.load_checkpoints()

        # Reset the environment
        state, _ = env.reset(seed=1234)

        # Render the initial environment state and store it
        env.render(ax)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.renderer.buffer_rgba()).copy())

        # Run the episode until the trained agent is done
        while not env.dones[trained_agent_id]:
            action = {}

            # Select deterministic action for all agents
            for ag in env.agents:
                action[ag.id] = agent.choose_action(
                    state=state[ag.id],
                    deterministic=True,
                )

            # Apply all actions and advance the environment by one step
            next_state, _, _, _, _ = env.step(action=action)

            # Replace the current observation with the next observation
            state = next_state

            # Render the updated environment state and store it
            env.render(ax)
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.renderer.buffer_rgba()).copy()
            frames.append(frame)

        print(f"[ DONE ] {exp.name}")

    # Close the figure
    plt.close(fig)

    # Save the animation if required
    if path is not None and file_name is not None:
        plot_animation(frames=frames, path=path, file_name=file_name, fps=fps)

        print("\n[ SAVE ] completed\n")

    return frames


def run_train_all(
    experiments: list[Experiment],
    previous_exp: Experiment | None,
    trained_agent_id: str,
    exp_name: str,
    n_steps: int = 200_000,
    chunk_steps: int = 2_000,
):
    """
    Train a single policy by alternating between multiple scenarios and log the training results.
    """

    # Fix the seed
    np.random.seed(1234)

    # Initialize the metrics and counter
    histories = []
    debugs = []
    total_steps = 0

    # Continue training until the requested total number of steps is reached
    while total_steps < n_steps:
        # Randomly select one scenario for the next training chunk
        env = random.choice(experiments)

        print(
            f"[ TRAIN MIXED ] "
            f"Step {total_steps:06d}/{n_steps:06d} | "
            f"Scenario = {env.name}"
        )

        # Limit the final chunk so the total step count does not exceed n_steps
        steps = min(chunk_steps, n_steps - total_steps)

        # Create the environment from the experiment configuration
        exp = Experiment(
            name=exp_name,
            env_config=env.env_config,
            agent_config=env.agent_config,
            reward_config=env.reward_config,
            n_steps=steps,
        )

        # Trained the policy
        history, debug, _ = run_sac(
            exp=exp,
            previous_exp=previous_exp,
            trained_agent_id=trained_agent_id,
            warmup_steps=0,
            reset_frequency=1,
        )

        # Record episode-level and debug metrics
        for row in history:
            row["scenario"] = env.name
        for row in debug:
            row["scenario"] = env.name

        # Accumulate metrics
        histories.extend(history)
        debugs.extend(debug)

        # Use the current experiment as initialization for the next chunk
        previous_exp = exp

        # Update the total steps count
        total_steps += steps

    # Reindex the episode numbre
    for i, history in enumerate(histories):
        history["episode"] = i
    for i, d in enumerate(debug):
        d["episode"] = i

    # Save training metrics
    df_train = log_train(metrics=histories, path="logs/train", file_name=exp_name)
    log_debug(metrics=debugs, path="logs/train", file_name=exp_name)
    df_rewards = log_rewards(metrics=histories, path="logs/train", file_name=exp_name)

    # Generate training figures
    plot_figures(
        df_perf=df_train,
        df_rewards=df_rewards,
        path="figures/train",
        file_name=exp_name,
    )


def run_train(
    experiments: list[Experiment],
    previous_exp: Experiment | None = None,
    trained_agent_id: str = "agent_1",
):
    """
    Train a policy for each experiment and save the corresponding logs and figures.
    """

    # Fix the seed
    np.random.seed(1234)

    # Record the training start time
    start = time.perf_counter()

    print(f"\n[ TRAIN ] {len(experiments)} experiment(s)\n")

    # Train all experiments
    for exp in experiments:
        # Train standard curriculum stages individually
        if "mixed_curriculum" not in exp.name:
            history, debug, _ = run_sac(
                exp=exp, previous_exp=previous_exp, trained_agent_id=trained_agent_id
            )

            # Save training metrics
            df_train = log_train(metrics=history, path="logs/train", file_name=exp.name)
            log_debug(metrics=debug, path="logs/train", file_name=exp.name)
            df_rewards = log_rewards(
                metrics=history, path="logs/train", file_name=exp.name
            )

            # Generate training figures
            plot_figures(
                df_perf=df_train,
                df_rewards=df_rewards,
                path="figures/train",
                file_name=exp.name,
            )

            # Use the current experiment as initialization for the next stage
            previous_exp = exp

        # Train mixed curriculum by alternating between experiments
        else:
            run_train_all(
                experiments=experiments,
                previous_exp=previous_exp,
                trained_agent_id=trained_agent_id,
                exp_name=exp.name,
            )

    # Compute the total training duration
    duration = time.perf_counter() - start

    print(
        f"\n[ SUMMARY ] {len(experiments)} experiment(s) completed in {duration:.1f}s\n"
    )


def run_validation(
    experiments: list[Experiment],
    n_episodes: int,
    n_render: int,
    trained_agent_id="agent_1",
):
    """
    Evaluate a trained policy on the validation scenarios and save the resulting metrics
    and animations.
    """

    # Fix the seed
    np.random.seed(1234)

    # Record the training start time
    start = time.perf_counter()

    print(f"\n[ VALIDATION ] {len(experiments)} experiment(s)\n")

    # Evaluate all experiments
    for exp in experiments:
        # Run the trained policy over multiple episodes
        history, frames = evaluate_sac(
            exp=exp,
            n_episodes=n_episodes,
            n_render=n_render,
            trained_agent_id=trained_agent_id,
        )

        # Run the trained policy over multiple episodes
        df_train = log_train(
            metrics=history, path="logs/validation", file_name=exp.name
        )
        df_rewards = log_rewards(
            metrics=history, path="logs/validation", file_name=exp.name
        )

        # Generate training figures
        plot_figures(
            df_perf=df_train,
            df_rewards=df_rewards,
            path="figures/validation",
            file_name=exp.name,
            window=100,
        )

        # Save an animation when rendering is enabled
        if n_render > 0:
            plot_animation(
                frames=frames, path="figures/validation", file_name=exp.name, fps=10
            )

    # Compute the total training duration
    duration = time.perf_counter() - start

    print(
        f"\n[ SUMMARY ] {len(experiments)} experiment(s) completed in {duration:.1f}s\n"
    )


def run_evaluation(
    experiments: list[Experiment],
    policy: Experiment,
    n_episodes: int = 100,
    n_render: int = 5,
    trained_agent_id: str = "agent_1",
):
    """
    Evaluate a trained policy on a set of scenarios and save the corresponding performance metrics and animations.
    """

    # Fix the seed
    np.random.seed(1234)

    # Record the training start time
    start = time.perf_counter()

    print(f"\n[ EVALUATION ] {len(experiments)} experiment(s)\n")

    # Evaluate all experiments with a chosen policy
    for scenario in experiments:
        # Create the environment from the experiment configuration
        exp = Experiment(
            name=policy.name,
            env_config=scenario.env_config,
            agent_config=scenario.agent_config,
            reward_config=scenario.reward_config,
            n_steps=scenario.n_steps,
        )

        # Run the trained policy over multiple episodes
        history, frames = evaluate_sac(
            exp=exp,
            n_episodes=n_episodes,
            n_render=n_render,
            trained_agent_id=trained_agent_id,
        )

        # Run the trained policy over multiple episodes
        df_train = log_train(
            metrics=history, path="logs/evaluation", file_name=scenario.name
        )
        df_rewards = log_rewards(
            metrics=history, path="logs/evaluation", file_name=scenario.name
        )

        # Generate training figures
        plot_figures(
            df_perf=df_train,
            df_rewards=df_rewards,
            path="figures/evaluation",
            file_name=exp.name,
            window=100,
        )

        # Save an animation when rendering is enabled
        if n_render > 0:
            plot_animation(
                frames=frames,
                path="figures/evaluation",
                file_name=scenario.name,
                fps=10,
            )

    # Compute the total training duration
    duration = time.perf_counter() - start

    print(
        f"\n[ SUMMARY ] {len(experiments)} experiment(s) completed in {duration:.1f}s\n"
    )
