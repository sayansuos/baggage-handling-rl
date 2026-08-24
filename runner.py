import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Literal

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from configs.config import Curriculum, Task
from rl.sac.agent import SACAgent
from rl.sac.sac import evaluate_sac, run_sac, set_checkpoint_paths
from simulator.environment.environment import Environment
from utils.logging import log_debug, log_metrics, log_rewards
from utils.plotting import plot_animation, plot_figures

# -------------------------------------------------------------------------------------
# MODE = train
# -------------------------------------------------------------------------------------


def run_train(
    tasks: list[Task],
    policy_name: str,
    n_trained_agents: int,
    init_policy_name: str | None,
    init_checkpoint_name: str | None,
    max_steps: float | None,
    sequential_curriculum: bool = False,
):
    """
    Train a SAC policy using either a fixed sequential curriculum
    or a probabilistic curriculum.
    """

    # Fix the seed
    np.random.seed(1234)

    # Record the training start time
    start = time.perf_counter()

    print(f"\n[ TRAIN ] {len(tasks)} task(s)\n")

    if sequential_curriculum:
        # -----------------------------------------------------------------------------
        # Fixed sequential curriculum
        # -----------------------------------------------------------------------------

        # Keep the same SAC agent across all curriculum stages
        agent = None

        for task in tasks:
            # Train the current curriculum stage
            _, _, agent = run_train_task(
                task=task,
                policy_name=policy_name,
                checkpoint_name=task.name,
                n_trained_agents=n_trained_agents,
                n_steps=task.n_steps,
                agent=agent,
                init_policy_name=init_policy_name,
                init_checkpoint_name=init_checkpoint_name,
                save_results=True,
                save_best=True,
            )

            init_policy_name = policy_name
            init_checkpoint_name = task.name

        set_checkpoint_paths(
            agent=agent, policy_name=policy_name, checkpoint_name="final"
        )
        agent.save_checkpoints()

    else:
        # -----------------------------------------------------------------------------
        # Probabilistic curriculum
        # -----------------------------------------------------------------------------

        # Get curriculum parameters from config
        chunk_steps = Curriculum.chunk_steps
        n_chunks = Curriculum.n_chunks
        threshold = Curriculum.threshold
        n_eval_episodes = Curriculum.n_eval_episodes

        run_curriculum(
            tasks=tasks,
            n_trained_agents=n_trained_agents,
            init_policy_name=init_policy_name,
            init_checkpoint_name=init_checkpoint_name,
            policy_name=policy_name,
            chunk_steps=chunk_steps,
            n_chunks=n_chunks,
            threshold=threshold,
            n_eval_episodes=n_eval_episodes,
            max_steps=max_steps,
        )

    # Compute the total training duration
    duration = time.perf_counter() - start

    print(f"\n[ SUMMARY ] {len(tasks)} task(s) completed in {duration:.1f}s\n")


def run_curriculum(
    tasks: list[Task],
    policy_name: str,
    n_trained_agents: int,
    init_policy_name: str | None,
    init_checkpoint_name: str | None,
    chunk_steps: int,
    n_chunks: int,
    threshold: float,
    n_eval_episodes: int,
    max_steps: float | None,
) -> None:
    """
    Train a SAC policy using a probabilistic curriculum based on
    a Dirichlet distribution.
    """

    # Number of curriculum tasks
    n_tasks = len(tasks)

    # Start with the easiest task
    focus_idx = 0

    # Name used to store the globally best curriculum policy
    best_policy_name = f"{policy_name}_best"

    # Latest evaluation success rate for every task
    success_rates = np.zeros(n_tasks)

    # Best average success rate observed across all tasks
    best_global_success = -np.inf

    # Keep one SAC agent during the entire curriculum
    agent = None

    # Set total budget : max_steps or same as fixed curriculum
    if max_steps is None:
        max_steps = sum(task.n_steps for task in tasks)

    episode_offset = 0
    total_steps = 0
    block = 0

    # Store all curriculum metrics
    all_history = []
    all_debug = []

    curriculum_completed = False

    # ---------------------------------------------------------------------------------
    # Curriculum loop
    # ---------------------------------------------------------------------------------

    while total_steps < max_steps:
        # Give the current focus task the largest Dirichlet parameter
        alpha = np.ones(n_tasks)
        alpha[focus_idx] = n_tasks

        print(
            f"\n[ CURRICULUM ] Focus: {tasks[focus_idx].name} | alpha={list(alpha)}\n"
        )

        # -----------------------------------------------------------------------------
        # Train several chunks before evaluation
        # -----------------------------------------------------------------------------
        for _ in range(n_chunks):
            if total_steps >= max_steps:
                break

            # Sample a probability vector over curriculum tasks
            weights = np.random.dirichlet(alpha)

            # Sample one task according to those probabilities
            task_idx = np.random.choice(
                n_tasks,
                p=weights,
            )
            task = tasks[task_idx]

            # Prevent exceeding the total training budget
            steps = int(min(chunk_steps, max_steps - total_steps))
            train_task = replace(task, n_steps=steps)

            print(
                f"[ CURRICULUM ] "
                f"Block {block} "
                f"| task={task.name} "
                f"| steps={steps} "
                f"| p={weights[task_idx]:.3f}"
            )

            # Train the SAC agent on the sampled task
            history, debug, agent = run_train_task(
                task=train_task,
                policy_name=policy_name,
                checkpoint_name="final",
                n_steps=steps,
                n_trained_agents=n_trained_agents,
                agent=agent,
                init_policy_name=init_policy_name,
                init_checkpoint_name=init_checkpoint_name,
                save_results=False,
                save_best=False,
            )

            init_policy_name = policy_name
            init_checkpoint_name = "final"

            # -------------------------------------------------------------------------
            # Add curriculum metrics
            # -------------------------------------------------------------------------

            chunk_episode_ids = [row["episode"] for row in history + debug]
            chunk_n_episodes = max(chunk_episode_ids, default=0)

            for row in history:
                row["task"] = task.name
                row["block"] = block
                row["episode"] += episode_offset

            for row in debug:
                row["task"] = task.name
                row["block"] = block
                row["episode"] += episode_offset
                row["step"] += total_steps

            episode_offset += chunk_n_episodes

            # Accumulate metrics across all chunks
            all_history.extend(history)
            all_debug.extend(debug)

            # Update curriculum counters
            total_steps += steps
            block += 1

        # -----------------------------------------------------------------------------
        # Evaluate the current policy on all tasks
        # -----------------------------------------------------------------------------

        n_workers = os.cpu_count() or 1

        print(
            "\n[ CURRICULUM ] Evaluating current policy... | "
            f"Number of workers: {n_workers}\n"
        )

        for i, task in enumerate(tasks):
            # Evaluate the SAC policy
            history, _, _, metrics = evaluate_sac(
                task=task,
                policy_name=policy_name,
                checkpoint_name="final",
                n_episodes=n_eval_episodes,
                n_renders=0,
                n_workers=n_workers,
                log_debug=False,
            )

            # Compute the success rate on this task
            success_rates[i] = np.mean([episode["success_rate"] for episode in history])

            print(
                f"[ {task.name} ] Evaluation terminated. "
                f"| Score = {success_rates[i]:.1%}"
                f"| Action time = {metrics['mean_action_time'] * 1000:.2f} ms "
                f"| Actions/s = {metrics['actions_per_second']:.1f} "
                f"| Actions = {metrics['n_actions']}",
                end="\r",
            )
            print()

        # Performance on the currently focused task
        focus_success = success_rates[focus_idx]
        # Global performance
        global_success = np.mean(success_rates)

        print(
            f"\n[ CURRICULUM ] "
            f"Focus task: {tasks[focus_idx].name} "
            f"| Focus success = {focus_success:.1%} "
            f"| Global success = {global_success:.1%} "
            f"| Threshold = {threshold:.1%}\n"
        )

        # --------------------------------------------------------------
        # Save the globally best curriculum policy
        # --------------------------------------------------------------

        if global_success > best_global_success:
            best_global_success = global_success

            # Redirect checkpoint paths toward "curriculum_best"
            set_checkpoint_paths(
                agent=agent, policy_name=best_policy_name, checkpoint_name="final"
            )
            agent.save_checkpoints()

            # Restore the paths used by the current curriculum policy
            set_checkpoint_paths(
                agent=agent, policy_name=policy_name, checkpoint_name="final"
            )

            print(f"[ CURRICULUM ] New best global success: {best_global_success:.1%}")

        # --------------------------------------------------------------
        # Update curriculum difficulty
        # --------------------------------------------------------------

        if focus_success >= threshold:
            # Move toward the next task
            if focus_idx < n_tasks - 1:
                focus_idx += 1

            else:
                # All tasks are mastered
                if np.min(success_rates) >= threshold:
                    print(
                        "\n[ CURRICULUM ] "
                        f"Curriculum completed "
                        f"| min success={np.min(success_rates):.1%}\n"
                    )
                    curriculum_completed = True

                else:
                    # Return to the task below the threshold
                    focus_idx = int(np.where(success_rates < threshold)[0][0])

        # --------------------------------------------------------------
        # Save cumulative curriculum logs
        # --------------------------------------------------------------

        # Save training metrics
        path = Path("logs/train") / policy_name
        log_metrics(metrics=all_history, path=path, file_name=policy_name)
        log_debug(metrics=all_debug, path=path, file_name=policy_name)
        log_rewards(metrics=all_history, path=path, file_name=policy_name)

        # Generate training figures
        figures = plot_figures(
            logs_path="logs",
            figs_path="figures",
            mode="train",
            policy_name=policy_name,
            file_name=policy_name,
            window=100,
        )
        for figure in figures:
            plt.close(figure)

        if curriculum_completed:
            break


def run_train_task(
    task: Task,
    policy_name: str,
    checkpoint_name: str,
    n_trained_agents: int,
    n_steps: int,
    agent: SACAgent | None = None,
    init_policy_name: str | None = None,
    init_checkpoint_name: str | None = None,
    save_results: bool = True,
    save_best: bool = True,
) -> tuple[list[dict], list[dict], SACAgent]:
    """
    Train the policy on a single task.
    """

    # Train the SAC policy
    history, debug, agent = run_sac(
        task=task,
        policy_name=policy_name,
        checkpoint_name=checkpoint_name,
        n_trained_agents=n_trained_agents,
        n_steps=n_steps,
        agent=agent,
        init_policy_name=init_policy_name,
        init_checkpoint_name=init_checkpoint_name,
        save_best=save_best,
    )

    # Save results when requested
    if save_results:
        # Save training metrics
        path = Path("logs/train") / policy_name
        log_metrics(metrics=history, path=path, file_name=task.name)
        log_debug(metrics=debug, path=path, file_name=task.name)
        log_rewards(metrics=history, path=path, file_name=task.name)

        # Generate training figures
        figures = plot_figures(
            logs_path="logs",
            figs_path="figures",
            mode="train",
            policy_name=policy_name,
            file_name=task.name,
            window=100,
        )
        for figure in figures:
            plt.close(figure)

    return history, debug, agent


# -------------------------------------------------------------------------------------
# MODE = validate
# -------------------------------------------------------------------------------------


def run_validation(
    tasks: list[Task],
    policy_name: str,
    checkpoint_strategy: Literal["matching", "final"],
    n_episodes: int,
    n_renders: int,
):
    """
    Evaluate a trained policy on the validation scenarios and save the resulting metrics
    and animations.
    """

    # Fix the seed
    np.random.seed(4321)

    # Use the maximum number of workers
    n_workers = os.cpu_count() or 1

    # Record the training start time
    start = time.perf_counter()

    print(
        f"\n[ VALIDATION ] {len(tasks)} task(s) | Number of workers: {n_workers} | "
        f"Episodes : {n_episodes} | Renders  : {n_renders}\n"
    )

    # Evaluate all tasks
    for task in tasks:
        print(f"[ {task.name} ] Evaluating... ", end="\r")

        # Run the trained policy over multiple episodes
        checkpoint_name = task.name if checkpoint_strategy == "matching" else "final"
        history, debug, frames, metrics = evaluate_sac(
            task=task,
            policy_name=policy_name,
            checkpoint_name=checkpoint_name,
            n_episodes=n_episodes,
            n_renders=n_renders,
            n_workers=n_workers,
            log_debug=True,
        )

        # Run the trained policy over multiple episodes
        path = Path("logs/validation") / policy_name
        log_metrics(metrics=history, path=path, file_name=task.name)
        log_debug(metrics=debug, path=path, file_name=task.name)
        log_rewards(metrics=history, path=path, file_name=task.name)

        # Save an animation when rendering is enabled
        if n_renders > 0:
            path = Path("figures/validation") / policy_name
            print(f"[ {task.name} ] Generating animation... ", end="\r")
            plot_animation(frames=frames, path=path, file_name=task.name, fps=10)

        success_rate = np.mean([episode["success_rate"] for episode in history])
        print(
            f"[ {task.name} ] Evaluation terminated. "
            f"| Success rate = {success_rate:.1%}"
            f"| Action time = {metrics['mean_action_time'] * 1000:.2f} ms "
            f"| Actions/s = {metrics['actions_per_second']:.1f} "
            f"| Actions = {metrics['n_actions']}",
            end="\r",
        )
        print()

    # Generate validation figures
    print("\n[ VALIDATION ] Generating figures... \n", end="\r")
    figures = plot_figures(
        logs_path="logs",
        figs_path="figures",
        mode="validation",
        policy_name=policy_name,
        file_name=policy_name,
        window=100,
    )
    for figure in figures:
        plt.close(figure)

    # Compute the total validation duration
    duration = time.perf_counter() - start

    print(f"\n[ SUMMARY ] {len(tasks)} task(s) completed in {duration:.1f}s\n")


# -------------------------------------------------------------------------------------
# MODE = evaluate
# -------------------------------------------------------------------------------------


def run_evaluation(
    tasks: list[Task],
    policy_name: str,
    n_episodes: int = 100,
    n_renders: int = 5,
):
    """
    Evaluate a trained policy on a set of tasks and save the corresponding
    performance metrics, figures and animations.
    """

    # Fix the seed
    np.random.seed(1234)

    # Record the start time
    start = time.perf_counter()

    # Use the maximum number of workers
    n_workers = os.cpu_count() or 1

    print(
        f"\n[ EVALUATION ] {len(tasks)} task(s) | Number of workers: {n_workers} | "
        f"Episodes : {n_episodes} | Renders  : {n_renders}\n"
    )

    # Evaluate all tasks with a chosen policy
    for task in tasks:
        print(f"[ {task.name} ] Evaluating... ", end="\r")

        # Run the selected policy over multiple episodes
        history, debug, frames, metrics = evaluate_sac(
            task=task,
            policy_name=policy_name,
            checkpoint_name="final",
            n_episodes=n_episodes,
            n_renders=n_renders,
            n_workers=n_workers,
            log_debug=True,
        )

        # Save training metrics
        print(f"[ {task.name} ] Saving metrics... ", end="\r")
        path = Path("logs/evaluation") / policy_name
        log_metrics(metrics=history, path=path, file_name=task.name)
        log_debug(metrics=debug, path=path, file_name=task.name)
        log_rewards(metrics=history, path=path, file_name=task.name)

        # Save an animation when rendering is enabled
        if n_renders > 0:
            print(f"[ {task.name} ] Generating animation... ", end="\r")
            path = Path("figures/evaluation") / policy_name
            plot_animation(frames=frames, path=path, file_name=task.name, fps=10)

        success_rate = np.mean([episode["success_rate"] for episode in history])
        print(
            f"[ {task.name} ] Evaluation terminated. "
            f"| Success rate = {success_rate:.1%}"
            f"| Action time = {metrics['mean_action_time'] * 1000:.2f} ms "
            f"| Actions/s = {metrics['actions_per_second']:.1f} "
            f"| Actions = {metrics['n_actions']}",
            end="\r",
        )
        print()

    # Generate training figures
    print("\n[ EVALUATION ] Generating figures... \n", end="\r")
    figures = plot_figures(
        logs_path="logs",
        figs_path="figures",
        mode="evaluation",
        policy_name=policy_name,
        file_name=policy_name,
        window=100,
    )
    for figure in figures:
        plt.close(figure)

    # Compute the total duration
    duration = time.perf_counter() - start

    print(f"\n[ SUMMARY ] {len(tasks)} task(s) completed in {duration:.1f}s\n")


# -------------------------------------------------------------------------------------
# MODE = animate
# -------------------------------------------------------------------------------------


def run_animation(
    tasks: list[Task],
    policy_name: str,
    checkpoint_name: str,
    file_name: str,
    fps: int,
) -> None:
    """
    Run one episode for each task and save the rendered frames as an animation.
    """

    print(f"\n[ ANIMATION ] {len(tasks)} task(s) | Policy: {policy_name}\n")

    # Initialize the frames
    frames = []

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Run one episode for each task
    for task in tasks:
        print(f"[ {task.name} ] Generating frames... ", end="\r")
        task_start_frame = len(frames)

        # Create the environment from the task configuration
        env = Environment(
            env_config=task.env_config,
            agent_config=task.agent_config,
            reward_config=task.reward_config,
            name=task.name,
        )

        # Define the number of focus agents
        env.set_focus_agents(n_focus_agents=task.env_config.nb_agents)

        # Load the selected trained policy
        agent = SACAgent(task=task, action_space=env.action_space)
        set_checkpoint_paths(
            agent=agent, policy_name=policy_name, checkpoint_name=checkpoint_name
        )
        agent.load_checkpoints()

        # Reset the environment
        state, _ = env.reset(seed=1234)

        # Render the initial environment state and store it
        env.render(ax)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.renderer.buffer_rgba()).copy())

        # Run the episode until all focus agents are done.
        while not env.done():
            action = {}

            # Select deterministic action for all agents
            for ag in env.agents:
                action[ag.id] = agent.choose_action(
                    state=state[ag.id], deterministic=True
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

        task_frames = len(frames) - task_start_frame
        print(f"[ {task.name} ] Frames generated | Frames: {task_frames}")

    # Close the figure
    plt.close(fig)

    # Save the animation
    print("\n[ ANIMATION ] Saving animation... ", end="\r")
    path = Path("figures/demo") / policy_name
    plot_animation(frames=frames, path=path, file_name=file_name, fps=fps)
    print(f"[ ANIMATION ] Animation saved | File: {path}/{file_name}_anim.mp4")

    print(
        f"\n[ SUMMARY ] {len(tasks)} task(s) rendered | Total frames: {len(frames)}\n"
    )


# -------------------------------------------------------------------------------------
# MODE = demo
# -------------------------------------------------------------------------------------


def run_demo(policy_name: str, file_name: str) -> None:
    """
    Run trained policies and interactively navigate through the rendered frames.
    """

    # Search for the path
    animation_path = Path("figures/demo") / policy_name / f"{file_name}_anim.mp4"

    if not animation_path.is_file():
        raise FileNotFoundError(f"Animation not found: {animation_path}")

    # Load the frames
    frames = [np.asarray(frame) for frame in iio.imiter(animation_path)]

    if not frames:
        raise ValueError(f"No frames found in: {animation_path}")

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
    animation = FuncAnimation(fig, update, interval=100, cache_frame_data=False)

    # Start paused
    animation.event_source.stop()

    # Display the first frame and start the interactive viewer
    render()
    plt.show()
