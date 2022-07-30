# How to write a game of snake in 12 lines of code



First of all a want to declare this article the founding document of snaketronics.

> Snaketronics:
>
> The scientific pursuit of the nature of computational snakes and all their
> implementations and applications in modern society

Then, secondly there is a disclosure. Before you start screaming
about line length, all of the lines are in fact PEP8 compliant.

We are also programming a version of Snake where the snake can loop around the
screen. However, you can change 2 lines to produce the original version of snake,
but I will leave that as an exercise for the reader.

We will be using `PyTorch` and `NumPy`. This
could have been done compleatly in either, but I prefer the `PyTorch` tensor
API and `NumPy` has a nice function called `unravel_index` that we will be using.

I will not be counting the imports and function declaration, call it freedom of
artistic expression, so I'll get that out of the way first. The code is also
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

Now that we've got all of the formalities out of the way; Buckle up Buckaroo,
because this is going to be a wild ride.

## Encodings

![Snaie Image](imgs/snake-encoding.drawio.svg)

The cruisal part of this code is the encoding of the snake state. As any computer
scientist worth his or her salt should know: "It's all about the data, baby!".

We want the encoding to be a matrix such that if displayed with `plt.imshow` it shows a
playable game of snake. It also needs to encode all of the information needed
to play snake in a manner that makes it easy to update.

For this reason we will encode the state as a matrix of integers where every open
cell in the game is 0, the tail of the snake is 1, every cell of the snake increases
with 1 towards the head and the food is -1. So; for a snake of size N, the tail will
be 1 and the head will be N.


Secondly we will be using a slightly odd encoding for actions. Instead of the traditional
`[up, right, down, left]` encoding we will be steeling the the action encoding from
the article [Teaching a computer how to play Snake with Q-Learning](
https://towardsdatascience.com/teaching-a-computer-how-to-play-snake-with-q-learning-93d0a316ddc0)
by Jason Lee. This encoding is `[left, straight, right]` relative to the current snake direction.
This is slightly harder to play for a human, but every action is always a valid action since
you can't go backwards.

# Code

## Getting current position

![Getting the positions](imgs/snake-get-pos.drawio.svg)

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

## Calculating the next position

![Next position diagram](imgs/snake-next-pos.drawio.svg)

In order to calculate the next position we do `pos_cur - pos_prev`. This yields the a vector
pointing in the current direction of travel of the snake. Next, we want to rotate it, but how much?

We want to rotate it `(270 + 90 * action)` degrees. This way when `0` is given as an action we turn left,
`1` we travel straight, and `2` turn right.

To achive this we apply a rotation matrix. If a matrix is applied to it self it gives us a matrix
that is equivalent to applying the transformation twice. Therefore, we can take the direction
vector and apply a 90 degree couter-clockwise rotation matrix raised to the power of `3 + action`.

Finally, we add the current position to this new direction vector to produce the next. Then we
take the new location, and pairwise mod it with the size of the board to generate the loop
around functionality.

```python
pos_next = (pos_cur - pos_prev) @ T([[0,-1],[1,0]]).matrix_power(3 + action)
pos_next = (pos_cur + pos_next) % T(snake.shape)
```
