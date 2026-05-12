import matplotlib.pyplot as plt
from simulator_env.environment import Environment

env = Environment(
    nb_agents=2,
    nb_static_obstacles=5,
    nb_moving_obstacles=2,
    env_width=70,
    env_height=50,
)

plt.ion()
env.render()
plt.show()
