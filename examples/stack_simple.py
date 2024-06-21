import gymnasium as gym
import panda_gym
from PIL import Image
import numpy as np

env_name = "PandaStackSimple-g-01x01-o1-02x02-o2-025x025-v3"
env = gym.make(env_name, render_mode="rgb_array")
observation, info = env.reset()

image = env.render()
image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
image_pil.save(r"{}.png".format(env_name))

env.close()