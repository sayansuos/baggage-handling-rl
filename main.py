from simulator_env.environment import Environment
import matplotlib.pyplot as plt

env = Environment(
    nb_agents=5,
    nb_static_obstacles=5,
    nb_moving_obstacles=2,
    env_width=70,
    env_height=50,
)

plt.ion()
env.render()
plt.show()
