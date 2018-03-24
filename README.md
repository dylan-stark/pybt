# PyBT

PyBT is a simple, extensible implementation of DeepMind's [Population Based Training of Neural Networks](https://deepmind.com/blog/population-based-training-neural-networks/).

## Install

```
pip install git+https://github.com/dylan-stark/pybt.git
```

## Play

If you just want to play around with it, you can use the included [Docker](https://docs.docker.com/install/) image that already includes Keras.
You can start a notebook server and run the [`notebooks/imdb.ipynb`]() notebook with

```
make -f docker/Makefile notebook
```

And you can run the examples from a bash shell with `make -f docker/Makefile bash`.

## Use

The primary usage mode is for training [Keras]() models.
This will typically invovle 1. wrapping your model and 2. initializing a trainer.
And the result is the Keras model with the best score, which you can use right away for scoring, etc.

```
from keras.models import Sequential
...

from pybt import Trainer
from pybt.model import KerasModelWrapper
from pybt.policy.done import StopAfter
from pybt.policy.ready import ReadyAfter

model = Sequential()
...

wrapped_model = KerasModelWrapper(model, optimizer=RMSprop(lr=0.001),
    loss=binary_crossentropy, metrics=['accuracy'])
trainer = Trainer(model=wrapped_model, stopping_criteria=StopAfter(20),
    ready_strategy=ReadyAfter(4),
    step_args = {'epochs_per_step': 2,
        'fit_args': {'x': partial_x_train, 'y': partial_y_train}},
    eval_args = {'x': x_val, 'y': y_val})
best_model, score = t.train()
```

Note that `KerasModelWrapper()` takes arbitrary compile options in order to support cloning the model.

Trainer initialization takes the wrapped model, various runtime strategies, and arguments to pass on to Keras's `fit()` and `evalaute()` methods.
The runtime strategies control the criteria for when the training session is done (e.g., after 4 epochs with `StopAfter(4)`), when a population member is ready for to exploit/explore (e.g., after each epoch with `ReadyAfter(1)`), and how to exploit the population (e.g., the default `Trucation()` which samples from the top 20% if the current member is in the bottom 20%).

