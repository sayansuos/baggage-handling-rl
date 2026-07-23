import numpy as np
import torch
from gymnasium import spaces

from configs.config import SACConfig, Task
from rl.sac import memory, networks


class SACAgent(torch.nn.Module):
    """
    Soft Actor-Critic agent responsible for action selection and network training.
    """

    def __init__(self, task: Task, action_space: spaces.Box):
        """
        Constructor
        """

        super(SACAgent, self).__init__()

        # Load the SAC configuration and store the task name
        self.sac_config = SACConfig()
        self.env_name = task.name

        # Shape of the local occupancy map observation
        self.map_shape = (
            task.agent_config.n_maps,
            task.agent_config.length_view,
            task.agent_config.length_view,
        )

        # Size of the non-spatial observation features
        self.obs_size = self.sac_config.obs_size

        # Size of the non-spatial observation features
        self.n_actions = action_space.shape[0]  # = 2 (v, omega)
        self.min_action = action_space.low
        self.max_action = action_space.high

        # SAC hyperparameters
        self.tau = self.sac_config.tau  # Soft update coefficient
        self.alpha = (
            self.sac_config.alpha
        )  # Entropy weight (taskloitation/taskloration)
        self.critic_lr = self.sac_config.critic_lr  # Critic learning rate
        self.actor_lr = self.sac_config.actor_lr  # Actor learning rate
        self.gamma = self.sac_config.gamma  # Discount factor
        self.reparam_noise = self.sac_config.reparam_noise  # Reparameter noise

        # Neural network and training parameters
        self.batch_size = self.sac_config.batch_size  # Number of transitions sampled
        self.feature_size = self.sac_config.feature_size  # Size of the latent vector
        self.hidden_size = self.sac_config.hidden_size  # Size of the hidden layers

        # Create the replay buffer
        self.mem_size = self.sac_config.mem_size  # Maximal size of the replay buffer
        self.memory = memory.ReplayBuffer(self.map_shape, self.n_actions, self.mem_size)

        # Create the two critic networks
        self.q1 = networks.CriticNetwork(
            self.map_shape,
            self.obs_size,
            self.feature_size,
            self.hidden_size,
            self.n_actions,
            self.critic_lr,
            chkpt_path=f"rl/SAC/weights/{self.env_name}_critic_1.pt",
        )
        self.q2 = networks.CriticNetwork(
            self.map_shape,
            self.obs_size,
            self.feature_size,
            self.hidden_size,
            self.n_actions,
            self.critic_lr,
            chkpt_path=f"rl/SAC/weights/{self.env_name}_critic_2.pt",
        )

        # Create the corresponding target networks
        self.target_q1 = networks.CriticNetwork(
            self.map_shape,
            self.obs_size,
            self.feature_size,
            self.hidden_size,
            self.n_actions,
            self.critic_lr,
            chkpt_path=f"rl/SAC/weights/{self.env_name}_target_critic_1.pt",
        )
        self.target_q2 = networks.CriticNetwork(
            self.map_shape,
            self.obs_size,
            self.feature_size,
            self.hidden_size,
            self.n_actions,
            self.critic_lr,
            chkpt_path=f"rl/SAC/weights/{self.env_name}_target_critic_2.pt",
        )

        # Create the actor network
        self.actor = networks.ActorNetwork(
            self.min_action,
            self.max_action,
            self.map_shape,
            self.obs_size,
            self.n_actions,
            self.feature_size,
            self.hidden_size,
            self.actor_lr,
            self.reparam_noise,
            chkpt_path=f"rl/SAC/weights/{self.env_name}_actor.pt",
        )

        # Initialize target critic networks with the critic network parameters
        self.update_network_params(tau=1)

    def choose_action(
        self,
        state: dict,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Select an action from the current policy for a given observation.
        """

        # Set the actor to evaluation mode
        self.actor.eval()

        # Convert each component of the observation to a batched PyTorch tensor
        # and move it to the device used by the actor network
        local_map = (
            torch.tensor(state["local_map"], dtype=torch.float32)
            .unsqueeze(0)
            .to(self.actor.device)
        )
        goal_relative_distance = (
            torch.tensor(state["goal_relative_distance"], dtype=torch.float32)
            .unsqueeze(0)
            .to(self.actor.device)
        )
        heading_error = (
            torch.tensor(state["heading_error"], dtype=torch.float32)
            .unsqueeze(0)
            .to(self.actor.device)
        )
        motion = (
            torch.tensor(state["motion"], dtype=torch.float32)
            .unsqueeze(0)
            .to(self.actor.device)
        )
        orientation = (
            torch.tensor(state["orientation"], dtype=torch.float32)
            .unsqueeze(0)
            .to(self.actor.device)
        )

        # Disable gradient computation
        with torch.no_grad():
            if deterministic:
                # Disable gradient computation
                mean, _ = self.actor.forward(
                    local_map,
                    goal_relative_distance,
                    heading_error,
                    motion,
                    orientation,
                )
                # Bound the action to [-1, 1]
                action = torch.tanh(mean)
                # Bound the action to [-1, 1]
                action = action * self.actor.action_scale + self.actor.action_bias

            else:
                # Sample a stochastic action from the current policy
                action, _ = self.actor.sample_normal(
                    local_map,
                    goal_relative_distance,
                    heading_error,
                    motion,
                    orientation,
                )

        # Restore the actor to training mode
        self.actor.train()

        return action.cpu().detach().numpy()[0]

    def store_transition(
        self,
        state: dict,
        action: np.ndarray,
        reward: float,
        next_state: dict,
        done: bool,
    ) -> None:
        """
        Store a transition in the replay buffer.
        """

        self.memory.store_transition(state, action, reward, next_state, done)

    def learn(self) -> None:
        """
        Perform one SAC update step.
        """

        # Wait until enough transitions are available to create a full mini-batch
        if self.memory.mem_counter < self.batch_size:
            return

        # Sample a random mini-batch of transitions from the replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        # Convert observations to tensors
        states = self._to_tensor_state(states)
        next_states = self._to_tensor_state(next_states)
        actions = torch.FloatTensor(actions).to(self.actor.device)
        rewards = torch.FloatTensor(rewards).to(self.actor.device)
        dones = torch.BoolTensor(dones).to(self.actor.device)

        # Disable gradient computation for target functions
        with torch.no_grad():
            # Sample the next actions from the current policy
            next_actions, next_log_probs = self.actor.sample_normal(
                next_states["local_map"],
                next_states["goal_relative_distance"],
                next_states["heading_error"],
                next_states["motion"],
                next_states["orientation"],
            )

            # Estimate the next-state Q-value with the target critics
            target_q1 = self.target_q1(
                next_states["local_map"],
                next_states["goal_relative_distance"],
                next_states["heading_error"],
                next_states["motion"],
                next_states["orientation"],
                next_actions,
            )
            target_q2 = self.target_q2(
                next_states["local_map"],
                next_states["goal_relative_distance"],
                next_states["heading_error"],
                next_states["motion"],
                next_states["orientation"],
                next_actions,
            )

            # Estimate the next-state Q-value with the first target critic
            target_q = torch.min(target_q1, target_q2)

            # Compute the entropy-regularized Bellman target
            # The future return is ignored when the transition is terminal
            q_hat = rewards.unsqueeze(1) + self.gamma * (
                1 - dones.float().unsqueeze(1)
            ) * (target_q - self.alpha * next_log_probs)

        # Compute the current Q-value predicted by the critics
        q1 = self.q1(
            states["local_map"],
            states["goal_relative_distance"],
            states["heading_error"],
            states["motion"],
            states["orientation"],
            actions,
        )
        q2 = self.q2(
            states["local_map"],
            states["goal_relative_distance"],
            states["heading_error"],
            states["motion"],
            states["orientation"],
            actions,
        )

        # Compute the current Q-value predicted by the first critic
        q1_loss = 0.5 * torch.nn.functional.mse_loss(q1, q_hat)
        q2_loss = 0.5 * torch.nn.functional.mse_loss(q2, q_hat)

        # Combine both critic losses
        critic_loss = q1_loss + q2_loss

        # Reset gradients accumulated by both critic optimizers
        self.q1.optimizer.zero_grad()
        self.q2.optimizer.zero_grad()

        # Compute gradients of the critic loss
        critic_loss.backward()

        # Update the parameters of both critic networks
        self.q1.optimizer.step()
        self.q2.optimizer.step()

        # Update the actor network
        self._actor_loss(states)

        # Update the actor network
        self.update_network_params()

    def _actor_loss(
        self,
        states: dict,
    ) -> None:
        """
        Compute and apply the SAC actor update.
        """

        # Sample new actions from the current policy
        new_actions, log_probs = self.actor.sample_normal(
            states["local_map"],
            states["goal_relative_distance"],
            states["heading_error"],
            states["motion"],
            states["orientation"],
        )

        # Evaluate the new actions with the critics
        q1_new = self.q1(
            states["local_map"],
            states["goal_relative_distance"],
            states["heading_error"],
            states["motion"],
            states["orientation"],
            new_actions,
        )
        q2_new = self.q2(
            states["local_map"],
            states["goal_relative_distance"],
            states["heading_error"],
            states["motion"],
            states["orientation"],
            new_actions,
        )

        # Keep the minimum Q-value
        q_new = torch.min(q1_new, q2_new)

        # Compute the SAC actor objective: minimize entropy term - taskected Q-value
        actor_loss = (self.alpha * log_probs - q_new).mean()

        # Reset gradients accumulated by the actor network
        self.actor.optimizer.zero_grad()

        # Compute gradients of the actor loss
        actor_loss.backward()

        # Update the parameters of the actor network
        self.actor.optimizer.step()

    def update_network_params(
        self,
        tau: float | None = None,
    ) -> None:
        """
        Soft-update target critic networks using Polyak averaging.
        """

        # Use the default soft-update coefficient if none is provided
        if tau is None:
            tau = self.tau

        # Update the parameters of the first target critic
        for target_param, param in zip(
            self.target_q1.parameters(),
            self.q1.parameters(),
        ):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        # Update the parameters of the second target critic
        for target_param, param in zip(
            self.target_q2.parameters(),
            self.q2.parameters(),
        ):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def save_checkpoints(self) -> None:
        """
        Save all network parameters to disk.
        """

        self.q1.save_checkpoint()
        self.q2.save_checkpoint()
        self.target_q1.save_checkpoint()
        self.target_q2.save_checkpoint()
        self.actor.save_checkpoint()

    def load_checkpoints(self) -> None:
        """
        Load all network parameters from disk.
        """

        self.q1.load_checkpoint()
        self.q2.load_checkpoint()
        self.target_q1.load_checkpoint()
        self.target_q2.load_checkpoint()
        self.actor.load_checkpoint()

    def _to_tensor_state(
        self,
        states: dict,
    ) -> dict[str, torch.Tensor]:
        """
        Convert a batch of observations from NumPy arrays
        to PyTorch tensors on the correct device.
        """

        # Convert each observation component to a float tensor on the correct device
        return {
            "local_map": torch.tensor(
                states["local_map"], dtype=torch.float32, device=self.actor.device
            ),
            "goal_relative_distance": torch.tensor(
                states["goal_relative_distance"],
                dtype=torch.float32,
                device=self.actor.device,
            ),
            "heading_error": torch.tensor(
                states["heading_error"], dtype=torch.float32, device=self.actor.device
            ),
            "motion": torch.tensor(
                states["motion"], dtype=torch.float32, device=self.actor.device
            ),
            "orientation": torch.tensor(
                states["orientation"], dtype=torch.float32, device=self.actor.device
            ),
        }
