# PyBT via Docker

This directory contains a [`Dockerfile`](https://docs.docker.com/) with the necessary tools for testing and experimenting with PyBT.

There's currently just one container with Keras and Tensorflow installed.
To start up the container with a [Jupyter notebook](http://jupyter.org/):

```
make notebook
```

And to start up the container with an interactive [iPython interpreter](https://ipython.org/) or Bash shell:

```
make ipython
make bash
```

In all cases, the current working directory will be mounted in the container at `/src/workspace` and if you have a `${HOME}/Data` directory it will be mounted at `/data`. Though both of these can be changed by overriding the `SRC` and `DATA` make variables.

