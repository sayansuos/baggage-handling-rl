import time
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from configs.config import Curriculum, Task
from rl.sac.agent import SACAgent
from rl.sac.sac import evaluate_sac, load_agent, run_sac, set_checkpoint_paths
from simulator.environment.environment import Environment
from utils.logging import log_debug, log_rewards, log_train
from utils.plotting import plot_animation, plot_figures


def run_validation(
    tasks: list[Task],
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

    print(f"\n[ VALIDATION ] {len(tasks)} task(s)\n")

    # Evaluate all tasks
    for task in tasks:
        # Run the trained policy over multiple episodes
        history, frames, _ = evaluate_sac(
            task=task,
            policy_name=task.name,
            n_episodes=n_episodes,
            n_render=n_render,
            trained_agent_id=trained_agent_id,
        )

        # Run the trained policy over multiple episodes
        df_train = log_train(
            metrics=history, path="logs/validation", file_name=task.name
        )
        df_rewards = log_rewards(
            metrics=history, path="logs/validation", file_name=task.name
        )

        # Generate training figures
        plot_figures(
            df_perf=df_train,
            df_rewards=df_rewards,
            path="figures/validation",
            file_name=task.name,
            window=100,
        )

        # Save an animation when rendering is enabled
        if n_render > 0:
            plot_animation(
                frames=frames, path="figures/validation", file_name=task.name, fps=10
            )

    # Compute the total training duration
    duration = time.perf_counter() - start

    print(f"\n[ SUMMARY ] {len(tasks)} task(s) completed in {duration:.1f}s\n")


def run_evaluation(
    tasks: list[Task],
    policy_name: str,
    n_episodes: int = 100,
    n_render: int = 5,
    trained_agent_id: str = "agent_1",
):
    """
    Evaluate a trained policy on a set of tasks and save the corresponding
    performance metrics, figures and animations.
    """

    # Fix the seed
    np.random.seed(1234)

    # Record the training start time
    start = time.perf_counter()

    print(f"\n[ EVALUATION ] {len(tasks)} task(s)\n")

    # Evaluate all tasks with a chosen policy
    for task in tasks:
        print(
            f"[ {task.name} ] Evaluating... ",
            end="\r",
        )

        # Run the selected policy over multiple episodes
        history, frames, metrics = evaluate_sac(
            task=task,
            policy_name=policy_name,
            n_episodes=n_episodes,
            n_render=n_render,
            trained_agent_id=trained_agent_id,
        )

        # Save training metrics
        print(
            f"[ {task.name} ] Saving metrics... ",
            end="\r",
        )
        df_train = log_train(
            metrics=history, path=f"logs/evaluation/{policy_name}", file_name=task.name
        )
        df_rewards = log_rewards(
            metrics=history, path=f"logs/evaluation/{policy_name}", file_name=task.name
        )

        # Generate training figures
        print(
            f"[ {task.name} ] Generating figures... ",
            end="\r",
        )
        plot_figures(
            df_perf=df_train,
            df_rewards=df_rewards,
            path=f"figures/evaluation/{policy_name}",
            file_name=task.name,
            window=100,
        )

        # Save an animation when rendering is enabled
        if n_render > 0:
            print(
                f"[ {task.name} ] Generating animation... ",
                end="\r",
            )
            plot_animation(
                frames=frames,
                path=f"figures/evaluation/{policy_name}",
                file_name=task.name,
                fps=10,
            )

        print(
            f"[ {task.name} ] Evaluation terminated "
            f"| Step time = {metrics['mean_step_time'] * 1000:.2f} ms "
            f"| FPS = {metrics['steps_per_second']:.1f} "
            f"| Steps = {metrics['n_steps']}",
            end="\r",
        )
        print()

    # Compute the total training duration
    duration = time.perf_counter() - start

    print(f"\n[ SUMMARY ] {len(tasks)} task(s) completed in {duration:.1f}s\n")


def run_train(
    tasks: list[Task],
    previous_task: Task | None = None,
    trained_agent_id: str = "agent_1",
    fixed_curriculum: bool = True,
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

    if fixed_curriculum:
        # --------------------------------------------------------------
        # Fixed sequential curriculum
        # --------------------------------------------------------------

        # Keep the same SAC agent across all curriculum stages
        agent = None

        for task in tasks:
            # Train the current curriculum stage
            _, _, agent = run_train_task(
                task=task,
                previous_task=previous_task,
                n_steps=task.n_steps,
                trained_agent_id=trained_agent_id,
                agent=agent,
                policy_name=task.name,
                save_results=True,
                save_best=True,
            )

            previous_task = task

    else:
        # --------------------------------------------------------------
        # Probabilistic curriculum
        # --------------------------------------------------------------

        # Get curriculum parameters from config
        chunk_steps = Curriculum.chunk_steps
        n_chunks = Curriculum.n_chunks
        threshold = Curriculum.threshold
        n_eval_episodes = Curriculum.n_eval_episodes

        run_curriculum(
            tasks=tasks,
            previous_task=previous_task,
            trained_agent_id=trained_agent_id,
            chunk_steps=chunk_steps,
            n_chunks=n_chunks,
            threshold=threshold,
            n_eval_episodes=n_eval_episodes,
        )

    # Compute the total training duration
    duration = time.perf_counter() - start

    print(f"\n[ SUMMARY ] {len(tasks)} task(s) completed in {duration:.1f}s\n")


def run_curriculum(
    tasks: list[Task],
    previous_task: Task | None,
    trained_agent_id: str,
    chunk_steps: int,
    n_chunks: int,
    threshold: float,
    n_eval_episodes: int,
) -> None:
    """
    Train a SAC policy using a probabilistic curriculum based on
    a Dirichlet distribution.
    """

    # Number of curriculum tasks
    n_tasks = len(tasks)

    # Start with the easiest task
    focus_idx = 0

    # Name of the curriculum policy
    policy_name = "curriculum"

    # Name used to store the globally best curriculum policy
    best_policy_name = "curriculum_best"

    # Latest evaluation success rate for every task
    scores = np.zeros(n_tasks)

    # Best average success rate observed across all tasks
    best_global_success = -np.inf

    # Keep one SAC agent during the entire curriculum
    agent = None

    # Used only when resuming training from a saved policy
    current_task = previous_task

    # Same total budget as the fixed curriculum
    max_steps = sum(task.n_steps for task in tasks)

    total_steps = 0
    block = 0

    # Store all curriculum metrics
    all_history = []
    all_debug = []

    # ---------------------------------------------------------
    # Curriculum loop
    # ---------------------------------------------------------

    while total_steps < max_steps:
        # Give the current focus task the largest Dirichlet parameter
        alpha = np.ones(n_tasks)
        alpha[focus_idx] = n_tasks

        print(
            f"\n[ CURRICULUM ] Focus: {tasks[focus_idx].name} | alpha={list(alpha)}\n"
        )

        # ---------------------------------------------------------
        # Train several chunks before evaluation
        # ---------------------------------------------------------
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
            steps = min(
                chunk_steps,
                max_steps - total_steps,
            )
            train_task = replace(
                task,
                n_steps=steps,
            )

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
                previous_task=current_task,
                n_steps=steps,
                trained_agent_id=trained_agent_id,
                agent=agent,
                policy_name=policy_name,
                save_results=False,
                save_best=False,
            )

            # ----------------------------------------------------------
            # Add curriculum metrics
            # ----------------------------------------------------------
            for row in history:
                row["task"] = task.name
                row["block"] = block
            for row in debug:
                row["task"] = task.name
                row["block"] = block

                # Convert the local chunk step into a global curriculum step
                row["step"] += total_steps

            # Accumulate metrics across all chunks
            all_history.extend(history)
            all_debug.extend(debug)

            # Update curriculum counters
            total_steps += steps
            block += 1

        # --------------------------------------------------------------
        # Evaluate the current policy on all tasks
        # --------------------------------------------------------------
        print("\n[ CURRICULUM ] Evaluating current policy...\n")

        for i, task in enumerate(tasks):
            # Evaluate the SAC policy
            history, _, _ = evaluate_sac(
                task=task,
                policy_name="curriculum",
                n_episodes=n_eval_episodes,
                n_render=0,
                trained_agent_id=trained_agent_id,
            )

            # Compute the success rate on this task
            scores[i] = np.mean([episode["success_rate"] for episode in history])
            print(f"{task.name}: {scores[i]:.1%}")

        focus_success = scores[focus_idx]  # Performance on the currently focused task
        global_success = np.mean(scores)  # Global performance

        print(
            f"\n[ CURRICULUM ] "
            f"Focus success = {focus_success:.1%} "
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
                agent=agent,
                policy_name=best_policy_name,
            )
            agent.save_checkpoints()

            # Restore the paths used by the current curriculum policy
            set_checkpoint_paths(
                agent=agent,
                policy_name=policy_name,
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
                if np.min(scores) >= threshold:
                    print(
                        "\n[ CURRICULUM ] "
                        f"Curriculum completed "
                        f"| min success={np.min(scores):.1%}\n"
                    )
                    break

                # Return to the task below the threshold
                focus_idx = int(np.where(scores < threshold)[0][0])

        # --------------------------------------------------------------
        # Save cumulative curriculum logs
        # --------------------------------------------------------------

        # Renumber episodes continuously
        for episode, history in enumerate(all_history, start=1):
            history["episode"] = episode
        for episode, debug in enumerate(all_debug, start=1):
            debug["episode"] = episode

        # Save training metrics
        df_train = log_train(
            metrics=all_history,
            path="logs/train",
            file_name="curriculum",
        )
        log_debug(
            metrics=all_debug,
            path="logs/train",
            file_name="curriculum",
        )
        df_rewards = log_rewards(
            metrics=all_history,
            path="logs/train",
            file_name="curriculum",
        )

        # Generate training figures
        plot_figures(
            df_perf=df_train,
            df_rewards=df_rewards,
            path="figures/train",
            file_name="curriculum",
        )


def run_train_task(
    task: Task,
    previous_task: Task | None,
    n_steps: int,
    trained_agent_id: str,
    agent: SACAgent | None = None,
    policy_name: str | None = None,
    save_results: bool = True,
    save_best: bool = True,
) -> tuple[list[dict], list[dict], SACAgent]:
    """
    Train the policy on a single task.
    """

    # Train the SAC policy
    history, debug, agent = run_sac(
        task=task,
        previous_task=previous_task,
        trained_agent_id=trained_agent_id,
        n_steps=n_steps,
        agent=agent,
        policy_name=policy_name,
        save_best=save_best,
    )

    # Save results when requested
    if save_results:
        # Save training metrics
        df_train = log_train(
            metrics=history,
            path="logs/train",
            file_name=task.name,
        )
        log_debug(
            metrics=debug,
            path="logs/train",
            file_name=task.name,
        )
        df_rewards = log_rewards(
            metrics=history,
            path="logs/train",
            file_name=task.name,
        )

        # Generate training figures
        plot_figures(
            df_perf=df_train,
            df_rewards=df_rewards,
            path="figures/train",
            file_name=task.name,
        )

    return history, debug, agent


def run_demo(
    tasks: list[Task],
    policy: Task,
    trained_agent_id: str = "agent_1",
) -> None:
    """
    Run trained policies and interactively navigate through the rendered frames.
    """

    # Generate all the scenarios frames
    frames = run_animation(
        tasks=tasks,
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
    tasks: list[Task],
    policy: Task,
    path: str | None = None,
    file_name: str | None = None,
    fps: int = 20,
    trained_agent_id: str = "agent_1",
) -> list[np.asarray]:
    """
    Run one episode for each task and return the rendered frames.
    If path and file_name are provided, also save the frames as an animation.
    """

    print(f"\n[ ANIMATION ] {len(tasks)} task(s)\n")

    # Initialize the frames
    frames = []

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Run one episode for each task
    for task in tasks:
        print(f"[ START ] {task.name}")

        # Create the environment from the task configuration
        env = Environment(
            env_config=task.env_config,
            agent_config=task.agent_config,
            reward_config=task.reward_config,
            name=task.name,
        )

        # Create the SAC agent using the selected trained policy
        agent = load_agent(env=env, task=policy, trained_agent_id=trained_agent_id)
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

        print(f"[ DONE ] {task.name}")

    # Close the figure
    plt.close(fig)

    # Save the animation if required
    if path is not None and file_name is not None:
        plot_animation(frames=frames, path=path, file_name=file_name, fps=fps)

        print("\n[ SAVE ] completed\n")

    return frames
