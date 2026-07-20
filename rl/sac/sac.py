import warnings

import matplotlib.pyplot as plt
import numpy as np

from configs.config import Experiment
from rl.sac.agent import SACAgent
from simulator.environment.environment import Environment

warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_agent(env: Environment, exp: Experiment, trained_agent_id: str) -> SACAgent:
    """
    Create a SAC agent with the observation and action dimensions of the environment.
    """

    agent = SACAgent(
        exp=exp,
        action_space=env.action_space[trained_agent_id],
    )

    return agent


def run_sac(
    exp: Experiment,
    previous_exp: Experiment | None,
    trained_agent_id: str,
    warmup_steps: int = 1000,
    update_frequency: int = 4,
    reset_frequency: int = 5,
    debug_frequency: int = 20,
) -> tuple[list[dict], list[dict], SACAgent]:
    """
    Train a SAC agent on an experiment, optionally initializing it from a previous policy.
    """

    # Create the training environment from the experiment configuration
    env = Environment(
        env_config=exp.env_config,
        agent_config=exp.agent_config,
        reward_config=exp.reward_config,
        name=exp.name,
    )

    # Create a new SAC agent when no previous experiment is provided
    if previous_exp is None:
        agent = load_agent(env=env, exp=exp, trained_agent_id=trained_agent_id)
    # Otherwise, initialize the agent from a previously trained policy
    else:
        agent = load_agent(env=env, exp=previous_exp, trained_agent_id=trained_agent_id)
        agent.load_checkpoints()
        agent.env_name = f"{exp.name}"
        agent.actor.checkpoint_path = f"rl/SAC/weights/{exp.name}_actor.pt"
        agent.q1.checkpoint_path = f"rl/SAC/weights/{exp.name}_critic_1.pt"
        agent.q2.checkpoint_path = f"rl/SAC/weights/{exp.name}_critic_2.pt"
        agent.target_q1.checkpoint_path = (
            f"rl/SAC/weights/{exp.name}_target_critic_1.pt"
        )
        agent.target_q2.checkpoint_path = (
            f"rl/SAC/weights/{exp.name}_target_critic_2.pt"
        )
        # Disable random warmup since the replay starts from an existing policy
        warmup_steps = 0

    # Initialize metrics, scores and counters
    history = []
    debug = []
    best_score = -np.inf
    scores = []
    total_steps = 0
    episode = 0

    # Disable random warmup since the replay starts from an existing policy
    while total_steps < exp.n_steps:
        episode += 1

        # Periodically regenerate the static obstacle configuration
        if episode % reset_frequency == 0:
            env.static_obstacles = env.env_manager.generate_static_obstacles()

        # Reset the environment
        state, _ = env.reset(seed=None)
        score = 0.0

        # Reset the environment
        if env.episode % debug_frequency == 0:
            env.debug = True
        else:
            env.debug = False

        # Run the episode until the trained agent is done
        # or the maximum training step count is reached.
        while not env.dones[trained_agent_id] and total_steps < exp.n_steps:
            action = {}

            # Select one action for each agent
            for ag in env.agents:
                # Apply the training policy to the selected agent
                if ag.id == trained_agent_id:
                    if total_steps < warmup_steps:  # Use random actions during warmup
                        action[ag.id] = env.action_space[trained_agent_id].sample()
                    else:  # Use stochastics actions from the learned policy afterward
                        action[ag.id] = agent.choose_action(
                            state=state[ag.id], deterministic=False
                        )

                # Control the other agents using the deterministic policy
                else:
                    action[ag.id] = agent.choose_action(
                        state=state[ag.id], deterministic=True
                    )

            # Apply all actions and advance the environment by one step
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

            # Update the SAC networks at the requested frequency
            if total_steps >= warmup_steps and total_steps % update_frequency == 0:
                agent.learn()

            # Store detailed metrics during debug episodes
            if env.debug:
                for agent_id, info in info.items():
                    debug.append(
                        {
                            "experiment": env.name,
                            "episode": episode,
                            "step": total_steps,
                            "agent": agent_id,
                            "pos_x": info["pos_x"],
                            "pos_y": info["pos_y"],
                            "distance_to_goal": info["distance_to_goal"],
                            "heading_error": info["heading_error"],
                            "min_obstacle_distance": info["min_obstacle_distance"],
                            "v": info["v"],
                            "omega": info["omega"],
                            "reward": reward[agent_id],
                            "reward_progress": info["reward_progress"],
                            "reward_rotation": info["reward_rotation"],
                            "reward_safety": info["reward_safety"],
                            "reward_collision": info["reward_collision"],
                            "state": info["state"],
                        }
                    )

        # Store episode-level metrics for non-debug episodes
        if not env.debug:
            history.append(info)

        # Update scores
        scores.append(score)
        avg_score = np.mean(scores[-100:])

        # Save the network parameters when a new best average is reached
        if avg_score > best_score:
            best_score = avg_score
            agent.save_checkpoints()

        # Display the current training progression
        print(
            f"[ {exp.name} ] Step {total_steps:06d}/{exp.n_steps} | Episode {episode:04d} | "
            f"Return = {score:8.3f} "
            f"Average = {avg_score:8.3f} "
            f"Best Average = {best_score:8.3f} "
            f"Success = {info['success_rate']:.0%} "
            f"Time = {info['mean_time_travel']:5.1f} ",
            end="\r",
        )

    print()

    return history, debug, agent


def evaluate_sac(
    exp: Experiment,
    n_episodes: int,
    n_render: int,
    trained_agent_id: str,
) -> tuple[list[dict], list[np.ndarray] | None]:
    """
    Evaluate a trained SAC agent over multiple episodes and optionally record rendered frames.
    """

    # Create the evaluation environment from the experiment configuration
    env = Environment(
        exp.env_config,
        exp.agent_config,
        exp.reward_config,
        exp.name,
    )

    # Load the trained agent using exp.name = policy
    agent = load_agent(env=env, exp=exp, trained_agent_id=trained_agent_id)
    agent.load_checkpoints()

    # Initialize metrics and frames
    history = []
    render = n_render > 0
    frames = [] if render else None

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Run the requested number of evaluation episodes
    for i in range(n_episodes):
        # Run the requested number of evaluation episodes
        if i == n_render:
            render = False
            plt.close(fig)

        # Generate a new static obstacle configuration for each episode
        env.static_obstacles = env.env_manager.generate_static_obstacles()

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

    return history, frames
