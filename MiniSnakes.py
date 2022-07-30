import torch as t
from torch import tensor as T
from numpy import unravel_index as unravel
import matplotlib.pyplot as plt

def do(snake, action):
    [pos_cur, pos_prev] = [T(unravel(x, snake.shape)) for x in snake.flatten().topk(2)[1]]
    pos_next = pos_cur + (pos_cur - pos_prev) @ T([[0,-1],[1,0]]).matrix_power(3 + action)
    pos_next = pos_next.clamp(T([0,0]), T(snake.shape) - 1)

    if (pos_cur == pos_next).all() or (snake[tuple(pos_next)] > 0).any():
        return (snake[tuple(pos_cur)] - 2).item()
    if snake[tuple(pos_next)] == -1:
        snake[unravel((snake == 0).flatten().to(t.float).multinomial(1)[0], snake.shape)] = -1
    else:
        snake[snake > 0] -= 1

    snake[tuple(pos_next)] = snake[tuple(pos_cur)] + 1

snake = t.zeros((64, 64), dtype=t.int)
snake[0,:3] = T([1, 2, -1])

fig, ax = plt.subplots(1,1)
img = ax.imshow(snake)
action = 1

def key_press(event):
    global action
    if event.key in ['a', 'd']:
        action = {'a': 0, 'd': 2}[event.key]
fig.canvas.mpl_connect('key_press_event', key_press)

while True:
    score = do(snake, action)
    if score is not None:
        print('Score:', score)
        break
    img.set_data(snake)
    fig.canvas.draw_idle()
    action = 1
    plt.pause(0.1)
