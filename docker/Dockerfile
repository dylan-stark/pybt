FROM ubuntu:16.04

ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN mkdir -p $CONDA_DIR && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh && \
    apt-get update && \
    apt-get install -y wget git libhdf5-dev g++ graphviz bzip2 && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh && \
    echo "c59b3dd3cad550ac7596e0d599b91e75d88826db132e4146030ef471bb434e9a *Miniconda3-4.2.12-Linux-x86_64.sh" | sha256sum -c - && \
    /bin/bash /Miniconda3-4.2.12-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-4.2.12-Linux-x86_64.sh

ENV NB_USER keras
ENV NB_UID 1000

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    mkdir -p $CONDA_DIR && \
    chown keras $CONDA_DIR -R && \
    mkdir -p /src && \
    chown keras /src

USER keras

# Python
ARG python_version=3.6

RUN conda install -y python=${python_version} && \
    pip install --upgrade pip && \
    pip install tensorflow && \
    conda install Pillow scikit-learn notebook numpy pandas matplotlib mkl nose pyyaml six h5py pytest && \
    pip install sklearn_pandas pytest-watch && \
    git clone git://github.com/keras-team/keras.git /src && pip install -e /src[tests] && \
    pip install git+git://github.com/keras-team/keras.git && \
    conda clean -yt

ENV KERAS_BACKEND=tensorflow

ENV PYTHONPATH='/src/:$PYTHONPATH'

WORKDIR /src/workspace/pybt

# Load MNIST data set
RUN python -c "from keras.datasets import imdb; imdb.load_data()"

RUN conda install -y python=${python_version} && \
    pip install git+git://github.com/dylan-stark/pybt.git  && \
    conda clean -yt

EXPOSE 8888

CMD jupyter notebook --port=8888 --ip=0.0.0.0

