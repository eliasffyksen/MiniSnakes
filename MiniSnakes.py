import torch as t
from torch import tensor as T
from numpy import unravel_index as unravel
import matplotlib.pyplot as plt


def do(snake: t.Tensor, action: int):
    positions = snake.flatten().topk(2)[1]
    [pos_cur, pos_prev] = [T(unravel(x, snake.shape)) for x in positions]
    rotation = T([[0, -1], [1, 0]]).matrix_power(3 + action)
    pos_next = (pos_cur + (pos_cur - pos_prev) @ rotation) % T(snake.shape)

    if (snake[tuple(pos_next)] > 0).any():
        return (snake[tuple(pos_cur)] - 2).item()
    if snake[tuple(pos_next)] == -1:
        pos_food = (snake == 0).flatten().to(t.float).multinomial(1)[0]
        snake[unravel(pos_food, snake.shape)] = -1
    else:
        snake[snake > 0] -= 1

    snake[tuple(pos_next)] = snake[tuple(pos_cur)] + 1
