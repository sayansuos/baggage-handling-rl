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
    init_task: Task | None,
    n_trained_agents: int,
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

    # ---------------------------------------------------------------------------------
    # Create the environment from the task configuration
    # ---------------------------------------------------------------------------------

    env = Environment(
        env_config=task.env_config,
        agent_config=task.agent_config,
        reward_config=task.reward_config,
        name=task.name,
    )

    # Define the number of trained agents
    env.set_focus_agents(n_focus_agents=n_trained_agents)

    # ---------------------------------------------------------------------------------
    # Initialize the SAC Agent
    # ---------------------------------------------------------------------------------

    # Case 1: We keep an agent replay buffer, so no warmup is required.
    if agent is not None:
        warmup_steps = 0

    # Case 2: We use the policy trained on a previous task to initialize the new one.
    elif init_task is not None:
        agent = SACAgent(task=init_task, action_space=env.action_space)
        agent.load_checkpoints()

    # Case 3: We build a new policy from scratch.
    else:
        agent = SACAgent(task=task, action_space=env.action_space)

    # Configure the checkpoint paths for the current policy.
    if policy_name is None:
        policy_name = task.name
    set_checkpoint_paths(agent=agent, policy_name=policy_name)

    # ---------------------------------------------------------------------------------
    # Initialize the metrics
    # ---------------------------------------------------------------------------------

    history = []
    debug = []

    best_success_rate = -np.inf
    success_rates = []

    total_steps = 0
    episode = 0

    # ---------------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------------

    # Run steps until the requested total number of steps is reached.
    while total_steps < n_steps:
        episode += 1

        # Periodically regenerate the static obstacle configuration.
        if episode % reset_frequency == 0:
            env.static_obstacles = env.env_manager.generate_static_obstacles()

        # Reset the environment.
        state, _ = env.reset(seed=None)

        # Enable detailed logging periodically.
        env.debug = env.episode % debug_frequency == 0

        while not env.done() and total_steps < n_steps:
            action = {}

            # -------------------------------------------------------------------------
            # Select actions for all agents
            # -------------------------------------------------------------------------

            for ag in env.agents:
                if ag in env.focus_agents:
                    # If the policy starts from scratch, use random actions during warmup
                    if init_task is None and total_steps < warmup_steps:
                        action[ag.id] = env.action_space.sample()

                    # Otherwise, use stochastics actions from the policy
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

            # -------------------------------------------------------------------------
            # Environment transition
            # -------------------------------------------------------------------------
            active_agents = tuple(ag for ag in env.focus_agents if not env.dones[ag.id])
            next_state, reward, _, _, info = env.step(action=action)

            # Store the active trained agent transitions in the replay buffer.
            for ag in active_agents:
                agent.store_transition(
                    state=state[ag.id],
                    action=action[ag.id],
                    reward=reward[ag.id],
                    next_state=next_state[ag.id],
                    done=env.dones[ag.id],
                )

            # Replace the current observation with the next observation
            state = next_state

            # Update counter
            total_steps += 1

            # -------------------------------------------------------------------------
            # SAC update
            # -------------------------------------------------------------------------

            # We only update the model once per step
            if total_steps >= warmup_steps and total_steps % update_frequency == 0:
                agent.learn()

            # -------------------------------------------------------------------------
            # Logging (debug and episode-level)
            # -------------------------------------------------------------------------

            # If debug, we store debug info at every step!
            if env.debug:
                for ag_id, ag_info in info.items():
                    debug.append(
                        {
                            "task": env.name,
                            "episode": episode,
                            "step": total_steps,
                            "agent": ag_id,
                            "pos_x": ag_info["pos_x"],
                            "pos_y": ag_info["pos_y"],
                            "distance_to_goal": ag_info["distance_to_goal"],
                            "heading_error": ag_info["heading_error"],
                            "min_obstacle_distance": ag_info["min_obstacle_distance"],
                            "v": ag_info["v"],
                            "omega": ag_info["omega"],
                            "reward": reward[ag_id],
                            "reward_progress": ag_info["reward_progress"],
                            "reward_rotation": ag_info["reward_rotation"],
                            "reward_safety": ag_info["reward_safety"],
                            "reward_collision": ag_info["reward_collision"],
                            "state": ag_info["state"],
                        }
                    )

        # -----------------------------------------------------------------------------
        # Episode-level logging
        # -----------------------------------------------------------------------------

        # Ignore the final episode if it has been interrupted.
        if not env.done():
            break

        # Store episode-level metrics
        if env.debug:
            episode_info = next(iter(info.values()))
        else:
            episode_info = info

        history.append(
            {
                "experiment": episode_info["experiment"],
                "episode": episode_info["episode"],
                "return_total": episode_info["return_total"],
                "mean_v": episode_info["mean_v"],
                "mean_abs_omega": episode_info["mean_abs_omega"],
                "success_rate": episode_info["success_rate"],
                "collision_rate": episode_info["collision_rate"],
                "mean_time_travel": episode_info["mean_time_travel"],
                "reward_progress": episode_info["reward_progress"],
                "reward_collision": episode_info["reward_collision"],
                "reward_safety": episode_info["reward_safety"],
                "reward_rotation": episode_info["reward_rotation"],
            }
        )

        # Average success_rate on the last 100 episodes.
        success_rate = episode_info["success_rate"]
        success_rates.append(success_rate)
        avg_success = np.mean(success_rates[-100:])

        # -----------------------------------------------------------------------------
        # Checkpoint selection
        # -----------------------------------------------------------------------------

        # For standard curriculum stages, retain the best policy according
        # to the rolling success rate once the window is fully populated.
        if save_best and len(success_rates) >= 100 and avg_success > best_success_rate:
            best_success_rate = avg_success
            agent.save_checkpoints()

        # Display the current average until a valid best value exists.
        best_display = (
            avg_success if best_success_rate == -np.inf else best_success_rate
        )

        # -----------------------------------------------------------------------------
        # Progress display
        # -----------------------------------------------------------------------------

        print(
            f"[ {task.name} ] "
            f"Step {total_steps:06d}/{n_steps} | "
            f"Episode {episode:04d} | "
            f"Return = {env.reward_total:8.3f} "
            f"Success = {success_rate:.0%} "
            f"Average Success = {avg_success:.1%} "
            f"Best Success = {best_display:.1%} "
            f"Time = {episode_info['mean_time_travel']:5.1f} ",
            end="\r",
        )

    # ---------------------------------------------------------------------------------
    # Final checkpoint
    # ---------------------------------------------------------------------------------

    # Probabilistic curriculum chunks retain their current policy.
    if not save_best or best_success_rate == -np.inf:
        agent.save_checkpoints()

    print()

    return history, debug, agent


def evaluate_sac(
    task: Task,
    policy_name: str,
    n_episodes: int,
    n_renders: int,
    n_workers: int,
) -> tuple[list[dict], list[np.ndarray] | None, dict]:
    """
    Evaluate a trained SAC policy on a task over multiple episodes.

    Rendered episodes are evaluated sequentially, while the remaining
    episodes are distributed across multiple processes.
    """

    # Split rendered and distributed episodes
    rendered_episode_ids = list(range(n_renders))
    parallel_episode_ids = list(range(n_renders, n_episodes))

    history = []
    frames = [] if n_renders > 0 else None
    action_times = []

    # ---------------------------------------------------------------------------------
    # Rendered episodes
    # ---------------------------------------------------------------------------------

    if rendered_episode_ids:
        render_history, frames, render_action_times = _evaluate_rendered_episodes(
            task=task,
            policy_name=policy_name,
            episode_ids=rendered_episode_ids,
        )

        history.extend(render_history)
        action_times.extend(render_action_times)

    # ---------------------------------------------------------------------------------
    # Distributed episodes
    # ---------------------------------------------------------------------------------

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
                executor.submit(_evaluate_worker, task, policy_name, episode_ids)
                for episode_ids in episode_chunks
            ]

            # Collect results
            for future in futures:
                worker_history, worker_action_times = future.result()
                history.extend(worker_history)
                action_times.extend(worker_action_times)

    # ---------------------------------------------------------------------------------
    # Restore episode order
    # ---------------------------------------------------------------------------------

    history.sort(key=lambda x: x[0])

    # Remove the episode indices
    history = [episode_info for _, episode_info in history]

    # ---------------------------------------------------------------------------------
    # Compute execution metrics
    # ---------------------------------------------------------------------------------

    metrics = {
        "n_actions": len(action_times),
        "mean_action_time": np.mean(action_times),
        "std_action_time": np.std(action_times),
        "actions_per_second": 1.0 / np.mean(action_times),
    }

    return history, frames, metrics


def _evaluate_worker(
    task: Task,
    policy_name: str,
    episode_ids: list[int],
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
    )

    history = []
    action_times = []

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

                action_times.append(time.perf_counter() - step_start)

            # Apply all actions and advance the environment by one step
            next_state, _, _, _, info = env.step(action=action)

            # Replace the current observation with the next observation
            state = next_state

        # Keep the episode index to restore the original order later
        info["episode"] = episode_id + 1
        history.append((episode_id, info))

    return history, action_times


def _evaluate_rendered_episodes(
    task: Task,
    policy_name: str,
    episode_ids: list[int],
) -> tuple[list[tuple[int, dict]], list[np.ndarray], list[float]]:
    """
    Evaluate and render a subset of episodes sequentially.
    """

    # Load the environment and policy
    env, agent = _load_evaluation_policy(
        task=task,
        policy_name=policy_name,
    )

    history = []
    frames = []
    action_times = []

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

                action_times.append(time.perf_counter() - step_start)

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

    return history, frames, action_times


def _load_evaluation_policy(
    task: Task,
    policy_name: str,
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

    # Evaluate all agents controlled by the shared policy.
    env.set_focus_agents(n_focus_agents=task.env_config.nb_agents)

    # Create a temporary task used only to load the selected policy
    policy_task = replace(
        task,
        name=policy_name,
    )

    # Load the trained agent
    agent = SACAgent(task=policy_task, action_space=env.action_space)
    agent.load_checkpoints()

    return env, agent
