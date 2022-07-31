import torch as t
from torch import tensor as T
import matplotlib.pyplot as plt
from MiniSnakes import do

snake = t.zeros((32, 32), dtype=t.int)
snake[0, :3] = T([1, 2, -1])

fig, ax = plt.subplots(1, 1)
img = ax.imshow(snake)
action = {'val': 1}
action_dict = {'a': 0, 'd': 2}
action_dict.setdefault(1)

fig.canvas.mpl_connect('key_press_event', lambda e:
                       action.__setitem__('val', action_dict[e.key]))

score = None
while score is None:
    img.set_data(snake)
    fig.canvas.draw_idle()
    plt.pause(0.1)
    score = do(snake, action['val'])
    action['val'] = 1

print('Score:', score)
