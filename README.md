# How to write a game of snake in 10 lines of code

First of all a want to declare this article the founding document of snaketronics.

> Snaketronics:
>
> The scientific pursuit of the nature of computational snakes and all their
> implementations and applications in modern society

Then, secondly there is a disclosure. We will be using PyTorch and NumPy. This
could have been done compleatly in either, but I prefer the PyTorch tensor
API and NumPy has a nice function called `unravel_index` that we will be using.
I will not be counting the imports and function declaration, so I'll get that out
of the way first.

```python
import torch as t
from torch import tensor as T
from numpy import unravel_index as unravel
import matplotlib.pyplot as plt

def do(snake, action):
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

!(imgs/snake-encoding.drawio.svg)

