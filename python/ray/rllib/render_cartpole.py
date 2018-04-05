import json
import numpy as np
import random
import os
from skimage.draw import line_aa, polygon
from scipy.misc import imsave


k = 0
if not os.path.exists("/tmp/cartpole"):
    os.makedirs("/tmp/cartpole")

BACKGROUND = np.tile(
    np.concatenate([np.arange(200, 0, -1), np.arange(0, 200, 1)]),
    (80, 1))

BACKGROUND_4XWIDE = np.tile(
    np.concatenate([
        np.arange(200, 0, -1),
        np.arange(0, 200, 1),
        np.arange(200, 0, -1),
        np.arange(0, 200, 1),
        np.arange(200, 0, -1),
        np.arange(0, 200, 1),
        np.arange(200, 0, -1),
        np.arange(0, 200, 1)]),
    (80, 1))

def render_frame(obs, env_config):
    cart_pos = obs[0]
    cart_velocity = obs[1]
    pole_angle = obs[2]
    angle_velocity = obs[3]
    angle_multiple = 3
    w = 80  # screen width
    xpos = cart_pos / 2.4 * w/2 + w/2
    if env_config["background"] == "noise":
        canvas = np.random.randint(200, size=(w, w), dtype=np.uint8)
    elif env_config["background"] == "zeros":
        canvas = np.zeros((w, w), dtype=np.uint8)
    elif env_config["background"] in ["fixed", "fixed_noisy"]:
        angle_multiple = 3
        xpos = w/2
        xs = int(np.clip(cart_pos / 2.4 * 160 + 160, 0, 2*200 - 80 - 1))
        xw = int(np.clip(cart_pos / 2.4 * 760 + 760, 0, 8*200 - 80 - 1))
        canvas = np.copy(BACKGROUND[:, xs:xs+80]).astype(np.float64)
        wide = np.copy(BACKGROUND_4XWIDE[:, xw:xw+80]).astype(np.float64)
        canvas[0:10,:] = wide[0:10,:]
        if env_config["background"] == "fixed_noisy":
            canvas = np.maximum(canvas, np.random.randint(200, size=(w, w)))
        canvas = canvas.astype(np.uint8)
    else:
        assert False, env_config

    canvas[0, :] = 100
    canvas[:, 0] = 100
    canvas[w-1, :] = 100
    canvas[:, w-1] = 100
    c = 5  # cart width
    left_aa_value = int((1 - xpos % 1) * 255)
    right_aa_value = int(xpos % 1 * 255)
#    print(xpos, xpos % 1, left_aa_value, right_aa_value)
    xpos = int(np.clip(np.ceil(xpos), 0, w-1))

    rr, cc = polygon(
        (1, w-1, w-1, 1),
        (w-12, w-12, w-1, w-1))
    canvas[cc, rr] = 0

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
            top_x = xpos + int(pole_length * np.sin(angle_multiple * pole_angle))
            top_y = w-10 - int(pole_length * np.cos(angle_multiple * pole_angle))
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

    for bg in ["zeros", "fixed", "fixed_noisy", "noise"]:
        print("testing", bg)
        for _ in range(10):
            render_frame([random.random() * 3] * 4, {"background": bg})

    prev = None
    ct = 0
    for i, line in enumerate(lines):
        canvas = render_frame(line["obs"], {"background": "fixed_noisy"}).squeeze()
        if prev == canvas.tolist():
            print("WARNING, similar obs", i)
            ct += 1
        prev = canvas.tolist()
        imsave(os.path.expanduser("~/Desktop/render/{}.png").format(k), canvas)
    print("Num warnings", ct)
