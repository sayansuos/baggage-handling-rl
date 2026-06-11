import numpy as np
import torch


class FeatureExtractor(torch.nn.Module):
    """
    Encode the observation into a latent feature vector.

    Observation:
    local_map, goal_relative_position, motion and orientation.
    """

    def __init__(
        self,
        map_shape: tuple,
        map_channels: int = 1,
        obs_size: int = 6,
        feature_size: int = 256,
    ):
        """
        Constructor
        """
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(map_channels, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(8, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2),
            # torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            # torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((4, 4)),
            torch.nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *map_shape)
            cnn_output_size = self.cnn(dummy).shape[1]  # 16 * 4 * 4 = 256

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(cnn_output_size + obs_size, feature_size),
            torch.nn.ReLU(),
        )
        self.feature_size = feature_size

    def forward(
        self,
        local_map,
        goal_relative_position,
        motion,
        orientation,
    ):
        map_features = self.cnn(local_map)
        obs_features = torch.cat(
            (
                goal_relative_position,
                motion,
                orientation,
            ),
            dim=1,
        )
        x = torch.cat(
            (
                map_features,
                obs_features,
            ),
            dim=1,
        )

        return self.fc(x)


class ActorNetwork(torch.nn.Module):
    def __init__(
        self,
        min_action: np.ndarray,
        max_action: np.ndarray,
        map_shape: tuple,
        n_actions: int = 2,
        feature_size: int = 256,
        hidden_size: int = 128,
        lr: float = 3e-5,
        reparam_noise=1e-6,
        chkpt_path="weights/actor.pt",
    ):
        """
        Constructor
        """

        super(ActorNetwork, self).__init__()

        self.lr = lr
        self.reparam_noise = reparam_noise
        self.checkpoint_path = chkpt_path

        self.features = FeatureExtractor(map_shape, feature_size=feature_size)

        self.actor = torch.nn.Sequential(
            torch.nn.Linear(feature_size, hidden_size),
            torch.nn.ReLU(),
            # torch.nn.Linear(hidden_size, hidden_size),
            # torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
        )

        self.mean = torch.nn.Linear(hidden_size, n_actions)
        self.log_std = torch.nn.Linear(hidden_size, n_actions)
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (max_action - min_action) / 2.0,
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "action_bias",
            torch.tensor(
                (max_action + min_action) / 2.0,
                dtype=torch.float32,
            ),
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), self.lr)

    def forward(
        self,
        local_map: torch.Tensor,
        goal_relative_position: torch.Tensor,
        motion: torch.Tensor,
        orientation: torch.Tensor,
    ):
        x = self.features(
            local_map,
            goal_relative_position,
            motion,
            orientation,
        )
        x = self.actor(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample_normal(
        self,
        local_map: torch.Tensor,
        goal_relative_position: torch.Tensor,
        motion: torch.Tensor,
        orientation: torch.Tensor,
    ):
        mean, log_std = self.forward(
            local_map,
            goal_relative_position,
            motion,
            orientation,
        )
        std = log_std.exp()
        probs = torch.distributions.Normal(mean, std)

        x_t = probs.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        scale = self.action_scale.to(y_t.device)
        bias = self.action_bias.to(y_t.device)

        action = y_t * scale + bias

        log_probs = probs.log_prob(x_t)
        log_probs -= torch.log(scale * (1 - y_t.pow(2)) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        # for deterministic policy return mu instead of action
        return action, log_probs

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_path)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_path))


class CriticNetwork(torch.nn.Module):
    def __init__(
        self,
        map_shape: tuple,
        feature_size: int = 256,
        hidden_size: int = 128,
        n_actions: int = 2,
        lr: float = 3e-4,
        chkpt_path: str = "weights/critic.pt",
    ):
        """
        Constructor
        """

        super(CriticNetwork, self).__init__()
        self.lr = lr
        self.checkpoint_path = chkpt_path

        self.features = FeatureExtractor(map_shape, feature_size=feature_size)

        self.critic = torch.nn.Sequential(
            torch.nn.Linear(feature_size + n_actions, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1),
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), self.lr)

    def forward(
        self,
        local_map: torch.Tensor,
        goal_relative_position: torch.Tensor,
        motion: torch.Tensor,
        orientation: torch.Tensor,
        action: torch.Tensor,
    ):
        x = self.features(
            local_map,
            goal_relative_position,
            motion,
            orientation,
        )
        x = torch.cat((x, action), dim=1)
        return self.critic(x)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_path)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_path))
