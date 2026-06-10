import torch

from rl.sac import memory, networks


class SACAgent(torch.nn.Module):
    def __init__(
        self,
        env_name,
        map_shape,
        action_space,
        tau=5e-3,
        alpha=0.2,
        batch_size=256,
        lr=3e-4,
        gamma=0.99,
        feature_size=256,
        hidden_size=128,
        mem_size=int(1e6),
    ):
        """
        Constructor
        """

        super(SACAgent, self).__init__()
        self.env_name = env_name
        self.map_shape = map_shape

        self.n_actions = action_space.shape[0]
        self.min_action = action_space.low
        self.max_action = action_space.high

        self.batch_size = batch_size
        self.tau = tau
        self.alpha = alpha
        self.lr = lr
        self.gamma = gamma
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.mem_size = mem_size

        self.memory = memory.ReplayBuffer(self.map_shape, self.n_actions, self.mem_size)

        self.q1 = networks.CriticNetwork(
            self.map_shape,
            self.feature_size,
            self.hidden_size,
            self.n_actions,
            self.lr,
            chkpt_path=f"rl/SAC/weights/{env_name}_critic_1.pt",
        )
        self.q2 = networks.CriticNetwork(
            self.map_shape,
            self.feature_size,
            self.hidden_size,
            self.n_actions,
            self.lr,
            chkpt_path=f"rl/SAC/weights/{env_name}_critic_2.pt",
        )

        self.target_q1 = networks.CriticNetwork(
            self.map_shape,
            self.feature_size,
            self.hidden_size,
            self.n_actions,
            self.lr,
            chkpt_path=f"rl/SAC/weights/{env_name}_target_critic_1.pt",
        )
        self.target_q2 = networks.CriticNetwork(
            self.map_shape,
            self.feature_size,
            self.hidden_size,
            self.n_actions,
            self.lr,
            chkpt_path=f"rl/SAC/weights/{env_name}_target_critic_2.pt",
        )

        self.actor = networks.ActorNetwork(
            self.min_action,
            self.max_action,
            self.map_shape,
            self.n_actions,
            self.feature_size,
            self.hidden_size,
            self.lr,
            chkpt_path=f"rl/SAC/weights/{env_name}_actor.pt",
        )

        self.update_network_params(tau=1)

    def choose_action(self, state: dict):
        """
        Select an action from the current policy for a given observation.
        """

        self.actor.eval()
        local_map = (
            torch.tensor(state["local_map"], dtype=torch.float32)
            .unsqueeze(0)
            .to(self.actor.device)
        )
        goal_relative_position = (
            torch.tensor(state["goal_relative_position"], dtype=torch.float32)
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
        with torch.no_grad():
            action, _ = self.actor.sample_normal(
                local_map, goal_relative_position, motion, orientation
            )
        self.actor.train()
        return action.cpu().detach().numpy()[0]

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer.
        """

        self.memory.store_transition(state, action, reward, next_state, done)

    def learn(self):
        """
        Perform one SAC update step.
        """

        if self.memory.mem_counter < self.batch_size:
            return

        # Observe the environment...
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )
        states = self._to_tensor_state(states)
        next_states = self._to_tensor_state(next_states)
        actions = torch.FloatTensor(actions).to(self.actor.device)
        rewards = torch.FloatTensor(rewards).to(self.actor.device)
        dones = torch.BoolTensor(dones).to(self.actor.device)

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample_normal(
                next_states["local_map"],
                next_states["goal_relative_position"],
                next_states["motion"],
                next_states["orientation"],
            )

            # Compute targets Q functions...
            target_q1 = self.target_q1(
                next_states["local_map"],
                next_states["goal_relative_position"],
                next_states["motion"],
                next_states["orientation"],
                next_actions,
            )

            target_q2 = self.target_q2(
                next_states["local_map"],
                next_states["goal_relative_position"],
                next_states["motion"],
                next_states["orientation"],
                next_actions,
            )
            target_q = torch.min(target_q1, target_q2)

            # Compute q_hat term i.e. scaled discounted returns...
            q_hat = rewards.unsqueeze(1) + self.gamma * (
                1 - dones.float().unsqueeze(1)
            ) * (target_q - self.alpha * next_log_probs)

        # Compute current Q functions...
        q1 = self.q1(
            states["local_map"],
            states["goal_relative_position"],
            states["motion"],
            states["orientation"],
            actions,
        )
        q2 = self.q2(
            states["local_map"],
            states["goal_relative_position"],
            states["motion"],
            states["orientation"],
            actions,
        )

        # Compute critic loss...
        q1_loss = 0.5 * torch.nn.functional.mse_loss(q1, q_hat)
        q2_loss = 0.5 * torch.nn.functional.mse_loss(q2, q_hat)
        critic_loss = q1_loss + q2_loss

        # Update...
        self.q1.optimizer.zero_grad()
        self.q2.optimizer.zero_grad()
        critic_loss.backward()
        self.q1.optimizer.step()
        self.q2.optimizer.step()

        self._actor_loss(states)

        self.update_network_params()

    def _actor_loss(self, states: dict):
        """
        Compute and apply the SAC actor update.
        """

        new_actions, log_probs = self.actor.sample_normal(
            states["local_map"],
            states["goal_relative_position"],
            states["motion"],
            states["orientation"],
        )

        # Get min critic value of states with current policy
        q1_new = self.q1(
            states["local_map"],
            states["goal_relative_position"],
            states["motion"],
            states["orientation"],
            new_actions,
        )
        q2_new = self.q2(
            states["local_map"],
            states["goal_relative_position"],
            states["motion"],
            states["orientation"],
            new_actions,
        )
        q_new = torch.min(q1_new, q2_new)

        # Compute actor loss...
        actor_loss = (self.alpha * log_probs - q_new).mean()

        # Update...
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

    def update_network_params(self, tau=None):
        """
        Soft-update target critic networks using Polyak averaging.
        """

        if tau is None:
            tau = self.tau

        for target_param, param in zip(
            self.target_q1.parameters(),
            self.q1.parameters(),
        ):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(
            self.target_q2.parameters(),
            self.q2.parameters(),
        ):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def save_checkpoints(self):
        """
        Save all network parameters to disk.
        """

        self.q1.save_checkpoint()
        self.q2.save_checkpoint()
        self.target_q1.save_checkpoint()
        self.target_q2.save_checkpoint()
        self.actor.save_checkpoint()

    def load_checkpoints(self):
        """
        Load all network parameters from disk.
        """

        self.q1.load_checkpoint()
        self.q2.load_checkpoint()
        self.target_q1.load_checkpoint()
        self.target_q2.load_checkpoint()
        self.actor.load_checkpoint()

    def _to_tensor_state(self, states: dict) -> dict:
        """
        Convert a batch of observations from NumPy arrays
        to PyTorch tensors on the correct device.
        """

        return {
            "local_map": torch.tensor(
                states["local_map"], dtype=torch.float32, device=self.actor.device
            ),
            "goal_relative_position": torch.tensor(
                states["goal_relative_position"],
                dtype=torch.float32,
                device=self.actor.device,
            ),
            "motion": torch.tensor(
                states["motion"], dtype=torch.float32, device=self.actor.device
            ),
            "orientation": torch.tensor(
                states["orientation"], dtype=torch.float32, device=self.actor.device
            ),
        }
