import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np
from simulator_env.environment import Environment

env = Environment(
    nb_agents=3,
    nb_static_obstacles=0,
    nb_moving_obstacles=5,
    env_width=60,
    env_height=80,
)

plt.ion()
for _ in range(100):
    env.step()
    env.render()
    plt.pause(0.01)
plt.ioff()


# ---------------------------------------------------------------
# To save a short video of the simulation
# ---------------------------------------------------------------

# writer = imageio.get_writer("anim.mp4", fps=30)

# for _ in range(300):
#     env.step()
#     env.render()
#     env.fig.canvas.draw()
#     frame = np.asarray(env.fig.canvas.renderer.buffer_rgba())
#     writer.append_data(frame)

# writer.close()
