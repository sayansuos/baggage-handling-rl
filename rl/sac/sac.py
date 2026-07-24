import time
import warnings
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

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
) -> tuple[list[dict], list[np.ndarray] | None, dict]:
    """
    Evaluate a trained SAC policy on a task over multiple episodes
    and optionally record rendered frames.
    """

    # Create the evaluation environment from the task configuration
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

    # Initialize metrics and frames
    history = []
    render = n_render > 0
    frames = [] if render else None
    step_times = []

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Run the requested number of evaluation episodes
    for i in range(n_episodes):
        step_start = time.perf_counter()

        # Run the requested number of evaluation episodes
        if i == n_render:
            render = False
            plt.close(fig)

        # Generate a new static obstacle configuration for each episode
        env.static_obstacles = env.env_manager.generate_static_obstacles()

        # Record execution time
        step_times.append(time.perf_counter() - step_start)

        # Reset the environment
        state, _ = env.reset(seed=1234 + i)

        # Record the initial environment state when rendering is enabled
        if render:
            env.render(ax)
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.renderer.buffer_rgba()).copy())

        # Record the initial environment state when rendering is enabled
        while not env.dones[trained_agent_id]:
            action = {}

            # Select deterministic action for all agents
            for ag in env.agents:
                action[ag.id] = agent.choose_action(
                    state=state[ag.id], deterministic=True
                )

            # Apply all actions and advance the environment by one step
            next_state, _, _, _, info = env.step(action=action)

            # Replace the current observation with the next observation
            state = next_state

            # Record the current environment frame when rendering is enabled
            if render:
                env.render(ax)
                fig.canvas.draw()
                frames.append(np.asarray(fig.canvas.renderer.buffer_rgba()).copy())

        # Store the metrics
        history.append(info)
        metrics = {
            "n_steps": len(step_times),
            "mean_step_time": np.mean(step_times),
            "std_step_time": np.std(step_times),
            "steps_per_second": 1.0 / np.mean(step_times),
        }

    return history, frames, metrics
