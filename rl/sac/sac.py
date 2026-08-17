import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import torch

from configs.config import Task
from rl.sac.agent import SACAgent
from simulator.environment.environment import Environment

warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_agent(env: Environment, task: Task, trained_agent_id: str) -> SACAgent:
    """
    Create a SAC agent with the observation and action dimensions of the environment.
    """

    agent = SACAgent(
        task=task,
        action_space=env.action_space[trained_agent_id],
    )

    return agent


def set_checkpoint_paths(agent: SACAgent, policy_name: str) -> None:
    """
    Configure all SAC checkpoint paths for a given policy name.
    """

    agent.env_name = policy_name

    agent.actor.checkpoint_path = f"rl/SAC/weights/{policy_name}_actor.pt"
    agent.q1.checkpoint_path = f"rl/SAC/weights/{policy_name}_critic_1.pt"
    agent.q2.checkpoint_path = f"rl/SAC/weights/{policy_name}_critic_2.pt"
    agent.target_q1.checkpoint_path = f"rl/SAC/weights/{policy_name}_target_critic_1.pt"
    agent.target_q2.checkpoint_path = f"rl/SAC/weights/{policy_name}_target_critic_2.pt"


def run_sac(
    task: Task,
    previous_task: Task | None,
    trained_agent_id: str,
    n_steps: int,
    agent: SACAgent | None = None,
    policy_name: str | None = None,
    warmup_steps: int = 1000,
    update_frequency: int = 4,
    reset_frequency: int = 5,
    debug_frequency: int = 20,
    save_best: bool = True,
) -> tuple[list[dict], list[dict], SACAgent]:
    """
    Train a SAC agent on a task.

    Three initialization cases are supported:
    - no agent and no previous task: start a new policy from scratch;
    - no agent but a previous task: load a previously trained policy;
    - existing agent: continue training while preserving the replay buffer.

    When save_best is True, the best policy is selected according to the
    rolling success rate over the last 100 episodes. Otherwise, the current
    policy is saved at the end of the training block.
    """

    # Create the training environment from the task configuration
    env = Environment(
        env_config=task.env_config,
        agent_config=task.agent_config,
        reward_config=task.reward_config,
        name=task.name,
    )

    # ---------------------------------------------------------
    # Initialize the SAC Agent
    # ---------------------------------------------------------

    if agent is not None:
        # Continue training with the existing agent and replay buffer
        warmup_steps = 0

    elif previous_task is not None:
        # Resume from a previously trained policy with a new replay buffer
        agent = load_agent(
            env=env,
            task=previous_task,
            trained_agent_id=trained_agent_id,
        )
        agent.load_checkpoints()

    else:
        # Start a new policy from scratch
        agent = load_agent(
            env=env,
            task=task,
            trained_agent_id=trained_agent_id,
        )

    # Configure the checkpoint paths for the current policy
    if policy_name is None:
        policy_name = task.name
    set_checkpoint_paths(agent=agent, policy_name=policy_name)

    # ---------------------------------------------------------
    # Initialize the metrics
    # ---------------------------------------------------------

    history = []
    debug = []

    best_success = -np.inf
    successes = []

    total_steps = 0
    episode = 0

    # ---------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------

    while total_steps < n_steps:
        episode += 1

        # Periodically regenerate the static obstacle configuration
        if episode % reset_frequency == 0:
            env.static_obstacles = env.env_manager.generate_static_obstacles()

        # Reset the environment
        state, _ = env.reset(seed=None)
        score = 0.0

        # Enable detailed logging periodically
        if env.episode % debug_frequency == 0:
            env.debug = True
        else:
            env.debug = False

        # Run the episode until the trained agent is done,
        # or the requested training budget is reached.
        while not env.dones[trained_agent_id] and total_steps < n_steps:
            action = {}

            # ---------------------------------------------------------
            # Select actions
            # ---------------------------------------------------------
            for ag in env.agents:
                if ag.id == trained_agent_id:
                    # If the policy starts from scratch, use random actions during warmup
                    if previous_task is None and total_steps < warmup_steps:
                        action[ag.id] = env.action_space[trained_agent_id].sample()

                    # Otherxise, use stochastics actions from the policy
                    else:
                        action[ag.id] = agent.choose_action(
                            state=state[ag.id],
                            deterministic=False,
                        )

                # Control the other agents using the deterministic policy
                else:
                    action[ag.id] = agent.choose_action(
                        state=state[ag.id], deterministic=True
                    )

            # ---------------------------------------------------------
            # Environment transition
            # ---------------------------------------------------------
            next_state, reward, _, _, info = env.step(action=action)

            # Store the trained agent transition in the replay buffer
            agent.store_transition(
                state=state[trained_agent_id],
                action=action[trained_agent_id],
                reward=reward[trained_agent_id],
                next_state=next_state[trained_agent_id],
                done=env.dones[trained_agent_id],
            )

            # Replace the current observation with the next observation
            state = next_state

            # Update metrics and counters
            score += reward[trained_agent_id]
            total_steps += 1

            # ---------------------------------------------------------
            # SAC update
            # ---------------------------------------------------------
            if total_steps >= warmup_steps and total_steps % update_frequency == 0:
                agent.learn()

            # ---------------------------------------------------------
            # Debug logging
            # ---------------------------------------------------------
            if env.debug:
                for agent_id, agent_info in info.items():
                    debug.append(
                        {
                            "task": env.name,
                            "episode": episode,
                            "step": total_steps,
                            "agent": agent_id,
                            "pos_x": agent_info["pos_x"],
                            "pos_y": agent_info["pos_y"],
                            "distance_to_goal": agent_info["distance_to_goal"],
                            "heading_error": agent_info["heading_error"],
                            "min_obstacle_distance": agent_info[
                                "min_obstacle_distance"
                            ],
                            "v": agent_info["v"],
                            "omega": agent_info["omega"],
                            "reward": reward[agent_id],
                            "reward_progress": agent_info["reward_progress"],
                            "reward_rotation": agent_info["reward_rotation"],
                            "reward_safety": agent_info["reward_safety"],
                            "reward_collision": agent_info["reward_collision"],
                            "state": agent_info["state"],
                        }
                    )

        # ---------------------------------------------------------
        # Episode-level logging
        # ---------------------------------------------------------

        # Get episode-level metrics
        if env.debug:
            # Debug info is stored separately for each agent
            episode_info = info[trained_agent_id]
            # Reconstruct the episode success metric for debug episodes
            episode_info["success_rate"] = (
                1.0 if "terminated" in episode_info["state"] else 0.0
            )

        else:
            # Standard info already contains episode-level metrics
            episode_info = info
            history.append(info)

        # Update the rolling success rate
        successes.append(episode_info["success_rate"])
        avg_success = np.mean(successes[-100:])

        # ---------------------------------------------------------
        # Checkpoint selection
        # ---------------------------------------------------------

        # For standard curriculum stages, retain the best policy according
        # to the rolling success rate once the window is fully populated.
        if len(successes) > 100 and avg_success > best_success:
            best_success = avg_success
            agent.save_checkpoints()

        # Display the current average until a valid best value exists
        best_display = avg_success if best_success == -np.inf else best_success

        # ---------------------------------------------------------
        # Progress display
        # ---------------------------------------------------------

        print(
            f"[ {task.name} ] "
            f"Step {total_steps:06d}/{n_steps} | "
            f"Episode {episode:04d} | "
            f"Return = {score:8.3f} "
            f"Success = {episode_info['success_rate']:.0%} "
            f"Average Success = {avg_success:.1%} "
            f"Best Success = {best_display:.1%} "
            f"Time = {episode_info['mean_time_travel']:5.1f} ",
            end="\r",
        )

    # ------------------------------------------------------------------
    # Final checkpoint
    # ------------------------------------------------------------------

    if save_best:
        # If fewer than 100 episodes were completed, retain the final policy.
        if best_success == -np.inf:
            agent.save_checkpoints()

    else:
        # Probabilistic curriculum chunks retain their current policy.
        agent.save_checkpoints()

    print()

    return history, debug, agent


def evaluate_sac(
    task: Task,
    policy_name: str,
    n_episodes: int,
    n_render: int,
    trained_agent_id: str,
    n_workers: int,
) -> tuple[list[dict], list[np.ndarray] | None, dict]:
    """
    Evaluate a trained SAC policy on a task over multiple episodes.

    Rendered episodes are evaluated sequentially, while the remaining
    episodes are distributed across multiple processes.
    """

    # Split rendered and distributed episodes
    rendered_episode_ids = list(range(n_render))
    parallel_episode_ids = list(range(n_render, n_episodes))

    history = []
    frames = [] if n_render > 0 else None
    step_times = []

    # ---------------------------------------------------------
    # Rendered episodes
    # ---------------------------------------------------------

    if rendered_episode_ids:
        render_history, frames, render_step_times = _evaluate_rendered_episodes(
            task=task,
            policy_name=policy_name,
            episode_ids=rendered_episode_ids,
            trained_agent_id=trained_agent_id,
        )

        history.extend(render_history)
        step_times.extend(render_step_times)

    # ---------------------------------------------------------
    # Distributed episodes
    # ---------------------------------------------------------

    if parallel_episode_ids:
        # Do not create more workers than episodes
        n_workers = min(n_workers, len(parallel_episode_ids))

        # Split episode between workers
        episode_chunks = [
            chunk.tolist()
            for chunk in np.array_split(parallel_episode_ids, n_workers)
            if len(chunk) > 0
        ]

        # Evaluate the chunks in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    _evaluate_worker, task, policy_name, episode_ids, trained_agent_id
                )
                for episode_ids in episode_chunks
            ]

            # Collect results
            for future in futures:
                worker_history, worker_step_times = future.result()
                history.extend(worker_history)
                step_times.extend(worker_step_times)

    # ---------------------------------------------------------
    # Restore episode order
    # ---------------------------------------------------------

    history.sort(key=lambda x: x[0])

    # Remove the episode indices
    history = [episode_info for _, episode_info in history]

    # ---------------------------------------------------------
    # Compute execution metrics
    # ---------------------------------------------------------

    metrics = {
        "n_actions": len(step_times),
        "mean_action_time": np.mean(step_times),
        "std_action_time": np.std(step_times),
        "actions_per_second": 1.0 / np.mean(step_times),
    }

    return history, frames, metrics


def _evaluate_worker(
    task: Task,
    policy_name: str,
    episode_ids: list[int],
    trained_agent_id: str,
) -> tuple[list[tuple[int, dict]], list[float]]:
    """
    Evaluate a subset of episodes in a separate process.
    """

    # Each worker uses a single PyTorch thread
    torch.set_num_threads(1)

    # Load one environment and one policy per worker
    env, agent = _load_evaluation_policy(
        task=task,
        policy_name=policy_name,
        trained_agent_id=trained_agent_id,
    )

    history = []
    step_times = []

    # Evaluate all episodes assigned to this worker
    for episode_id in episode_ids:
        seed = 1234 + episode_id
        np.random.seed(seed)

        # Generate a new static obstacle configuration
        env.static_obstacles = env.env_manager.generate_static_obstacles()

        # Reset the environment
        state, _ = env.reset(seed=seed)

        # Run the episode until completion
        while not env.done():
            action = {}
            # Select deterministic actions for all agents
            for ag in env.agents:
                # Measure the action choice execution time
                step_start = time.perf_counter()

                action[ag.id] = agent.choose_action(
                    state=state[ag.id],
                    deterministic=True,
                )

                step_times.append(time.perf_counter() - step_start)

            # Apply all actions and advance the environment by one step
            next_state, _, _, _, info = env.step(action=action)

            # Replace the current observation with the next observation
            state = next_state

        # Keep the episode index to restore the original order later
        info["episode"] = episode_id + 1
        history.append((episode_id, info))

    return history, step_times


def _evaluate_rendered_episodes(
    task: Task,
    policy_name: str,
    episode_ids: list[int],
    trained_agent_id: str,
) -> tuple[list[tuple[int, dict]], list[np.ndarray], list[float]]:
    """
    Evaluate and render a subset of episodes sequentially.
    """

    # Load the environment and policy
    env, agent = _load_evaluation_policy(
        task=task,
        policy_name=policy_name,
        trained_agent_id=trained_agent_id,
    )

    history = []
    frames = []
    step_times = []

    # Create the rendering figure
    fig, ax = plt.subplots(figsize=(10, 8))

    for episode_id in episode_ids:
        seed = 1234 + episode_id
        np.random.seed(seed)

        # Generate a new static obstacle configuration
        env.static_obstacles = env.env_manager.generate_static_obstacles()

        # Reset the environment
        state, _ = env.reset(seed=seed)

        # Render the initial state
        env.render(ax)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.renderer.buffer_rgba()).copy())

        # Run the episode until completion
        while not env.done():
            action = {}

            # Select deterministic actions for all agents
            for ag in env.agents:
                # Measure the action choice execution time
                step_start = time.perf_counter()

                action[ag.id] = agent.choose_action(
                    state=state[ag.id],
                    deterministic=True,
                )

                step_times.append(time.perf_counter() - step_start)

            # Apply all actions and advance the environment by one step
            next_state, _, _, _, info = env.step(action=action)

            # Replace the current observation with the next observation
            state = next_state

            # Render the current state
            env.render(ax)
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.renderer.buffer_rgba()).copy())

        # Keep the episode index to restore the original order later
        info["episode"] = episode_id + 1
        history.append((episode_id, info))

    plt.close(fig)

    return history, frames, step_times


def _load_evaluation_policy(
    task: Task,
    policy_name: str,
    trained_agent_id: str,
) -> tuple[Environment, SACAgent]:
    """
    Create an evaluation environment and load the selected trained policy.
    """

    # Create the evaluation environment
    env = Environment(
        env_config=task.env_config,
        agent_config=task.agent_config,
        reward_config=task.reward_config,
        name=task.name,
    )

    # Create a temporary task used only to load the selected policy
    policy_task = replace(
        task,
        name=policy_name,
    )

    # Load the trained agent
    agent = load_agent(
        env=env,
        task=policy_task,
        trained_agent_id=trained_agent_id,
    )
    agent.load_checkpoints()

    return env, agent
