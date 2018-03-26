import json
import numpy as np
import random
import os
from skimage.draw import line_aa, polygon
from scipy.misc import imsave


def render_frame(obs):
    cart_pos = obs[0]
    cart_velocity = obs[1]
    pole_angle = obs[2]
    angle_velocity = obs[3]
    canvas = np.zeros((42, 42), dtype=np.uint8)

    canvas[0, :] = 100
    canvas[:, 0] = 100
    canvas[41, :] = 100
    canvas[:, 41] = 100
    xpos = int(cart_pos * 21 + 21)
    rr, cc = polygon(
        (max(xpos - 5, 0), min(xpos + 5, 41),
         min(xpos + 5, 41), max(xpos - 5, 0)),
        (35, 35, 40, 40))
    canvas[cc, rr] = 255

    error = None
    for pole_length in [30, 20, 10, 5, 0]:
        try:
            top_x = xpos + int(pole_length * np.sin(pole_angle))
            top_y = 34 - int(pole_length * np.cos(pole_angle))
            rr, cc, val = line_aa(
                top_x, top_y, xpos, 34)
            canvas[cc, rr] = val * 255
            error = None
            break
        except Exception as e:
            error = e
    if error:
        raise error

    return np.expand_dims(canvas, 2)


if __name__ == '__main__':
    lines = []
    for line in open(os.path.expanduser("~/Desktop/cartpole-random.json")).readlines():
        lines.append(json.loads(line))
        if len(lines) > 1000:
            break

    for k, line in enumerate(lines):
        canvas = render_frame(line["obs"]).squeeze()
        imsave(os.path.expanduser("~/Desktop/render/{}.png").format(k), canvas)
