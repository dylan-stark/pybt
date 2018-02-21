FROM tensorflow/tensorflow:1.5.0-py3

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    unzip \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    python3-numpy \
    python3-scipy \
    python3-matplotlib \
    python3-yaml \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    libhdf5-serial-dev \
    python3-h5py \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    graphviz \
    python3-pydot \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    python3-setuptools

RUN \
    git clone https://github.com/fchollet/keras && \
    cd keras && \
    python3 setup.py install

RUN \
    git clone https://github.com/pytest-dev/pytest.git && \
    cd pytest && \
    python3 setup.py install

RUN groupadd -r pybt && useradd -r -m -g pybt pybt
USER pybt

ENV KERAS_BACKEND=tensorflow

WORKDIR /src

EXPOSE 8888

CMD jupyter notebook --port=8888 --ip=0.0.0.0 --allow-root

