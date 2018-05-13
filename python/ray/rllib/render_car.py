import json
import numpy as np
import os
from scipy.misc import imsave
from ray.rllib.utils.compression import unpack
from ray.rllib.render_cartpole import create_snow_dynamic
import random

# Fix Python 2.x.
try:
    UNICODE_EXISTS = bool(type(unicode))
except NameError:
    unicode = lambda s: str(s)

num_rendered = 0

def add_car_snow(obs, num_snow=20, noise=True):
    global num_rendered
    num_rendered += 1
    if num_rendered % 10000 == 0:
        print("Num rendered", num_rendered)
    if type(obs) in [str, unicode]:
        obs = unpack(obs)
    if noise:
        noise_vec = np.random.randint(100, size=(80, 80, 1), dtype=np.uint8)
        obs = obs / 2
        obs = obs + noise_vec
    if num_snow:
        obs = create_snow_dynamic(num_snow, obs, 9, snow_size=7, intensities=[0, 200])
        if random.random() < .001:
            imsave("/tmp/snow_sample-{}.png".format(num_rendered), obs.squeeze())
    return obs

if __name__ == '__main__':
    lines = []
    for line in open(os.path.expanduser("~/Desktop/car-repeat.json")).readlines():
        lines.append(json.loads(line))
        if len(lines) > 1000:
            break

    ct = 0
    for k, line in enumerate(lines):
        canvas = add_car_snow(line["obs"])[:, :, -1].squeeze()
        imsave(os.path.expanduser("~/Desktop/render_car/{}.png").format(k), canvas)
