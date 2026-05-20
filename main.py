import numpy as np
import matplotlib.pyplot as plt
from simulator.utils.config import EnvConfig, AgentConfig
from simulator.environment.environment import Environment
from simulator.utils.save_figures import save_grid, save_animation

np.random.seed(123)

env_config = EnvConfig()
agent_config = AgentConfig()
env = Environment(env_config, agent_config)

save_grid(env.grid_map.current_grid, "grid.png", scale=10, show_grid=True)
save_grid(
    env._get_local_grid(env.agents[0]), "grid_local.png", scale=50, show_grid=True
)
save_animation(env, "anim.mp4")

plt.ion()
for _ in range(100):
    env.step()
    env.render()
    plt.pause(0.01)
plt.ioff()
