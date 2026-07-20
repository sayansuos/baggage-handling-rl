import numpy as np
import torch


class FeatureExtractor(torch.nn.Module):
    """
    Encode the observation into a latent feature vector.

    Observation:
    local_map, goal_relative_distance, heading_error, motion and orientation.
    """

    def __init__(
        self,
        map_shape: tuple,
        obs_size: int,
        feature_size: int,
    ):
        """
        Constructor
        """

        super().__init__()

        # Build the convolutional encoder
        self.cnn = torch.nn.Sequential(
            # Extract spatial features from the stacked local maps
            torch.nn.Conv2d(map_shape[0], 4, kernel_size=3, padding=1),
            # Extract spatial features from the stacked local maps
            torch.nn.ReLU(),
            # Reduce each feature map to a fixed 3 x 3 spatial representation
            torch.nn.AdaptiveAvgPool2d((3, 3)),
            # Flatten the convolutional output into a 1D vector
            torch.nn.Flatten(),
        )

        # Determine the size of the convolutional feature vector automatically
        with torch.no_grad():
            dummy = torch.zeros(1, *map_shape)
            cnn_output_size = self.cnn(dummy).shape[1]  # 4 * 3 * 3 = 36

        # Determine the size of the convolutional feature vector automatically
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(cnn_output_size + obs_size, feature_size),
            torch.nn.ReLU(),
        )

        # Store the size of the resulting latent feature vector.
        self.feature_size = feature_size

    def forward(
        self,
        local_map: torch.Tensor,
        goal_relative_distance: torch.Tensor,
        heading_error: torch.Tensor,
        motion: torch.Tensor,
        orientation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode the complete observation into a latent feature vector.
        """

        # Extract spatial features from the local occupancy maps
        map_features = self.cnn(local_map)

        # Concatenate all observation components, except occupancy maps
        obs_features = torch.cat(
            (
                goal_relative_distance,
                heading_error,
                motion,
                orientation,
            ),
            dim=1,
        )

        # Concatenate all observation components
        x = torch.cat(
            (
                map_features,
                obs_features,
            ),
            dim=1,
        )

        # Project the observation into the latent feature space
        x = self.fc(x)

        return x


class ActorNetwork(torch.nn.Module):
    def __init__(
        self,
        min_action: np.ndarray,
        max_action: np.ndarray,
        map_shape: tuple,
        obs_size: int,
        n_actions: int,
        feature_size: int,
        hidden_size: int,
        lr: float,
        reparam_noise: float,
        chkpt_path: str,
    ):
        """
        Constructor
        """

        super(ActorNetwork, self).__init__()

        # Store the checkpoint file path
        self.checkpoint_path = chkpt_path

        # Store the optimization and numerical stability parameters
        self.lr = lr
        self.reparam_noise = reparam_noise

        # Create the shared feature extractor for the observation
        self.features = FeatureExtractor(
            map_shape=map_shape,
            obs_size=obs_size,
            feature_size=feature_size,
        )

        # Build the fully connected actor backbone
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(feature_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
        )

        # Predict the mean of the Gaussian action distribution
        self.mean = torch.nn.Linear(hidden_size, n_actions)

        # Predict the logarithm of the Gaussian standard deviation
        self.log_std = torch.nn.Linear(hidden_size, n_actions)

        # Store the scale used to map actions from [-1, 1] to the environment bounds
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (max_action - min_action) / 2.0,
                dtype=torch.float32,
            ),
        )

        # Store the bias used to center actions within the environment bounds
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (max_action + min_action) / 2.0,
                dtype=torch.float32,
            ),
        )

        # Select the device
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.to(self.device)

        # Create the Adam optimizer used to update the actor parameters
        self.optimizer = torch.optim.Adam(self.parameters(), self.lr)

    def forward(
        self,
        local_map: torch.Tensor,
        goal_relative_distance: torch.Tensor,
        heading_error: torch.Tensor,
        motion: torch.Tensor,
        orientation: torch.Tensor,
    ):
        """
        Encode the complete observation into a latent feature vector.
        """

        # Encode the observation into a latent feature vector
        x = self.features(
            local_map,
            goal_relative_distance,
            heading_error,
            motion,
            orientation,
        )

        # Process the latent features through the actor backbone
        x = self.actor(x)

        # Compute the mean of the Gaussian distribution
        mean = self.mean(x)

        # Compute the logarithm of the standard deviation
        log_std = self.log_std(x)

        # Compute the logarithm of the standard deviation
        log_std = torch.clamp(log_std, -20, 2)

        # Return the Gaussian distribution parameters
        return mean, log_std

    def sample_normal(
        self,
        local_map: torch.Tensor,
        goal_relative_distance: torch.Tensor,
        heading_error: torch.Tensor,
        motion: torch.Tensor,
        orientation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample a continuous action from the actor policy.
        """

        # Compute the Gaussian distribution parameters
        mean, log_std = self.forward(
            local_map=local_map,
            goal_relative_distance=goal_relative_distance,
            heading_error=heading_error,
            motion=motion,
            orientation=orientation,
        )

        # Convert the logarithmic standard deviation to standard deviation
        std = log_std.exp()

        # Create the Gaussian action distribution
        probs = torch.distributions.Normal(mean, std)

        # Sample using the reparameterization trick (mean + std * N(0,1))
        x_t = probs.rsample()

        # Bound the sampled values to [-1, 1]
        y_t = torch.tanh(x_t)

        # Retrieve the action scaling parameters (scale and bias)
        scale = self.action_scale.to(y_t.device)
        bias = self.action_bias.to(y_t.device)

        # Rescale the normalized action to the environment action bounds
        action = y_t * scale + bias

        # Compute the log-probability of the sampled action, correct it and rescale it
        log_probs = probs.log_prob(x_t)
        log_probs -= torch.log(scale * (1 - y_t.pow(2)) + self.reparam_noise)

        # Sum the log-probabilities over all action dimensions
        log_probs = log_probs.sum(1, keepdim=True)

        # Return the sampled action and its corrected log-probability
        return action, log_probs

    def save_checkpoint(self) -> None:
        """
        Save the actor network parameters.
        """

        torch.save(self.state_dict(), self.checkpoint_path)

    def load_checkpoint(self) -> None:
        """
        Load the actor network parameters.
        """

        # Load the saved parameters on the current computation device
        state_dict = torch.load(
            self.checkpoint_path,
            map_location=self.device,
        )

        # Restore the network parameters
        self.load_state_dict(state_dict)

        # Ensure the network remains on the selected device
        self.to(self.device)


class CriticNetwork(torch.nn.Module):
    def __init__(
        self,
        map_shape: tuple,
        obs_size: int,
        feature_size: int,
        hidden_size: int,
        n_actions: int,
        lr: float,
        chkpt_path: str,
    ):
        """
        Constructor
        """

        super(CriticNetwork, self).__init__()

        # Store the checkpoint file path
        self.checkpoint_path = chkpt_path

        # Store the optimization parameter
        self.lr = lr

        # Create the shared feature extractor for the observation
        self.features = FeatureExtractor(
            map_shape=map_shape,
            obs_size=obs_size,
            feature_size=feature_size,
        )

        # Build the fully connected critic backbone
        self.critic = torch.nn.Sequential(
            # Combine the encoded observation with the action
            torch.nn.Linear(feature_size + n_actions, hidden_size),
            torch.nn.ReLU(),
            # Process the state-action representation
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            # Output a single Q-value
            torch.nn.Linear(hidden_size, 1),
        )

        # Select the device
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.to(self.device)

        # Create the Adam optimizer used to update the actor parameters
        self.optimizer = torch.optim.Adam(self.parameters(), self.lr)

    def forward(
        self,
        local_map: torch.Tensor,
        goal_relative_distance: torch.Tensor,
        heading_error: torch.Tensor,
        motion: torch.Tensor,
        orientation: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate the Q-value of an observation-action pair.
        """

        # Encode the observation into a latent feature vector
        x = self.features(
            local_map,
            goal_relative_distance,
            heading_error,
            motion,
            orientation,
        )

        # Concatenate the encoded observation with the selected action
        x = torch.cat((x, action), dim=1)

        # Estimate and the corresponding Q-value
        x = self.critic(x)

        return x

    def save_checkpoint(self) -> None:
        """
        Save the critic network parameters.
        """

        torch.save(self.state_dict(), self.checkpoint_path)

    def load_checkpoint(self) -> None:
        """
        Load the critic network parameters.
        """

        # Load the saved parameters on the current computation device
        state_dict = torch.load(
            self.checkpoint_path,
            map_location=self.device,
        )

        # Restore the network parameters
        self.load_state_dict(state_dict)

        # Ensure the network remains on the selected device
        self.to(self.device)
