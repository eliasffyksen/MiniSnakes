# How to write a game of snake in 10 lines of code

First of all a want to declare this article the founding document of snaketronics.

> Snaketronics:
>
> The scientific pursuit of the nature of computational snakes and all their
> implementations and applications in modern society

Then, secondly there is a disclosure. We will be using `PyTorch` and `NumPy`. This
could have been done compleatly in either, but I prefer the `PyTorch` tensor
API and `NumPy` has a nice function called `unravel_index` that we will be using.
I will not be counting the imports and function declaration (call it freedom of
artistic expression), so I'll get that out of the way first. The code is also
not very readable, sensible or propper in any way, shape or form. But, sometimes
it's important to not write code that's "correct", but code that is fun.

```python
import torch as t
from torch import tensor as T
from numpy import unravel_index as unravel
import matplotlib.pyplot as plt

def do(snake, action):
    '''This is where the magic happens :) '''
```

Now that we've got everythhing set up; Buckle up Buckaroo, because this is going
to be a wild ride.

The cruisal part of this code is the encoding of the snake state. As any computer
scientist worth his or her salt should know: "It's all in the data, baby!".

We want the encoding to be a matrix such that if displayed with `plt.imshow` it shows a
playable game of snake. It also needs to encode all of the information needed
to play snake in a manner that makes it easy to update. Therefore we will encode the state
as a matrix of integers where every open cell in the game is 0, the tail of the snake is 1,
every cell of the snake increases with 1 towards the head and the food is -1. So; for a
snake of size N, the tail will be 1 and the head will be N. The image below shows an
illustration of the encoding:

![Snaie Image](imgs/snake-encoding.drawio.svg)

Secondly we will be using a slightly odd encoding for actions. Instead of the traditional
`[up, right, down, left]` encoding we will be steeling the the action encoding from
the article [Teaching a computer how to play Snake with Q-Learning](
https://towardsdatascience.com/teaching-a-computer-how-to-play-snake-with-q-learning-93d0a316ddc0)
by Jason Lee. This encoding is `[left, straight, right]` relative to the current snake direction.
This is slightly harder to play for a human, but every action is always a valid action since
you can't go backwards.

Now, with the formalities out of the way, we can get pick up where we left off and get into the
trenches of actually writing code. The first thing to do is to get the current and previous
possition of the snake head. We can do this with the `topk(2)`, since the head of the snake
is always the largest int and the previous head is the second largest. The only problem we have
is that the `topk` method works along one dimention at a time. For this reason we need to `flatten()`
the tensor first, get the topk then use the forementioned `unravel_index` to convert it back
to a 2d index. At last we want to turn them into tensors so that we can do math on them as well.

```python
[pos_cur, pos_prev] = [T(unravel(x, snake.shape)) for x in snake.flatten().topk(2)[1]]
```

