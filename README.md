# PyBT

PyBT is a simple, extensible implementation of DeepMind's [Population Based Training of Neural Networks](https://deepmind.com/blog/population-based-training-neural-networks/).

## Install

```
pip install git+https://github.com/dylan-stark/pybt.git
```

## Usage

The primary usage mode is for training [Keras]() models.
This will typically invovle wrapping your model and initializing a trainer.
The result is the Keras model with the best score, which you can use right away for inferencing, etc.

A good way to get started is to try it out for yourself.
You can use the included [Docker](https://docs.docker.com/install/) image, which already includes Keras, to start a notebook server and run the [`notebooks/imdb.ipynb`]() notebook with

```
make -f docker/Makefile notebook
```

You can also run the examples from a bash shell with `make -f docker/Makefile bash`.

The following snippet shows a basic setup.

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
The runtime strategies control the criteria for when the training session is done (e.g., after 4 epochs with `StopAfter(4)`), when a population member is ready to exploit/explore (e.g., after each epoch with `ReadyAfter(1)`), and how to exploit the population (e.g., the default `Trucation()` which samples from the top 20% if the current member is in the bottom 20%).

