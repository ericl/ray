import json
import numpy as np
import random
import os
from skimage.draw import line_aa, polygon
from scipy.misc import imsave


k = 0
if not os.path.exists("/tmp/cartpole"):
    os.makedirs("/tmp/cartpole")

def render_frame(obs):
    cart_pos = obs[0]
    cart_velocity = obs[1]
    pole_angle = obs[2]
    angle_velocity = obs[3]
    w = 80  # screen width
    canvas = np.zeros((w, w), dtype=np.uint8)

    canvas[0, :] = 100
    canvas[:, 0] = 100
    canvas[w-1, :] = 100
    canvas[:, w-1] = 100
    xpos = cart_pos / 2.4 * w/2 + w/2
    c = 5  # cart width
    left_aa_value = int((1 - xpos % 1) * 255)
    right_aa_value = int(xpos % 1 * 255)
#    print(xpos, xpos % 1, left_aa_value, right_aa_value)
    xpos = int(np.clip(np.ceil(xpos), 0, w-1))

    # draw antialiased border
    c2 = 6  # wider by a little for antialiasing
    rr, cc = polygon(
        (max(0, xpos - c2), xpos, xpos, max(0, xpos - c2)),
        (w-10, w-10, w-5, w-5))
    canvas[cc, rr] = left_aa_value
    rr, cc = polygon(
        (xpos, min(w-1, xpos + c2), min(w-1, xpos + c2), xpos),
        (w-10, w-10, w-5, w-5))
    canvas[cc, rr] = right_aa_value

    rr, cc = polygon(
        (max(0, xpos - c), min(w-1, xpos + c), min(w-1, xpos + c), max(0, xpos - c)),
        (w-10, w-10, w-5, w-5))
    canvas[cc, rr] = 255

    error = None
    for pole_length in [60, 30, 20, 10, 5, 0]:
        try:
            top_x = xpos + int(pole_length * np.sin(pole_angle))
            top_y = w-10 - int(pole_length * np.cos(pole_angle))
            rr, cc, val = line_aa(
                top_x, top_y, xpos, w-10)
            canvas[cc, rr] = val * 255
            error = None
            break
        except Exception as e:
            error = e
    if error:
        raise error

    global k
    k += 1
    k %= 1000
#    imsave("/tmp/cartpole/{}.png".format(k), canvas)

    return np.expand_dims(canvas, 2)


if __name__ == '__main__':
    lines = []
    for line in open(os.path.expanduser("~/Desktop/cartpole-expert.json")).readlines():
        lines.append(json.loads(line))
        if len(lines) > 1000:
            break

    prev = None
    for i, line in enumerate(lines):
        canvas = render_frame(line["obs"]).squeeze()
        if prev == canvas.tolist():
            print("WARNING, similar obs", i)
        prev = canvas.tolist()
        imsave(os.path.expanduser("~/Desktop/render/{}.png").format(k), canvas)
