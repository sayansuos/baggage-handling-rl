import matplotlib.pyplot as plt
from simulator.environment.environment import Environment
from simulator.utils import *

env = Environment(
    nb_agents=1,
    nb_static_obstacles=10,
    nb_moving_obstacles=5,
    env_width=100,
    env_height=60,
)

save_grid(env.grid_map.current_grid, "grid.png")
save_grid(env.get_local_grids()[0], "grid_local.png")
save_animation(env, "anim.mp4")

plt.ion()
for _ in range(1):
    env.step()
    env.render()
    plt.pause(5)
plt.ioff()
