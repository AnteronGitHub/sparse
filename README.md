# Edge Deep Learning

This repository contains source code for experiments about Deep learning in the Edge.

## Compatibility

The software has been tested, and thus can be considered compatible with, the following devices and the following
software:

| Device            | JetPack version | Python version | PyTorch version | Docker version | Docker file suffix |
| ----------------- | --------------- | -------------- | --------------- | -------------- | ------------------ |
| Jetson AGX Xavier | 5.0 preview     | 3.8.10         | 1.12.0a0        | 20.10.12       | jp50               |
| Jetson Nano 2GB   | 4.6.1           | 3.6.9          | 1.9.0           | 20.10.7        | jp461              |
| Lenovo ThinkPad   | -               | 3.8.12         | 1.11.0          | 20.10.15       | amd64              |

## Install

The repository uses PyTorch as the primary Deep Learning framework. Software dependencies can be installed with pip or
by using Docker.

### Python

For running on common amd64 processor architecture, public PyPi distributions can be used. Install PyTorch by running:

```
pip install torch
```

On client nodes install torchvision, by running:

```
pip install torchvision
```

*NB! On Jetson devices, PyTorch has to be installed by using pip wheels provided by NVIDIA. See
[NVIDIA forum](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-11-now-available/72048) for more
details.*

### Docker

Software dependencies are included in Docker images which can be built locally. Run the following command, replacing
\<node\> with the node that is being built (client/server), and \<arch\> with the compatible architecture or Jetson SDK
(see Docker suffix in the compatibility table above).

```
docker build . -f dockerfiles/Dockerfile.split_<node>.<arch> -t edge-deep-learning-code:split_<node>.<arch>
```

## Run program

### Python

With dependencies installed locally, agents can be started with the following command, by replacing \<node\> with the
corresponding node (client/server):

```
python3 src/split_training_<node>.py
```

### Docker

With docker images built according to the above instructions, new containers can be created by running the following
command, replacing \<node\> and \<arch\> as when building the images:

```
docker run --rm -it edge-deep-learning-code:split_<node>.<arch>
```

## Configure

Nodes can be configured by setting the following environment variables. Parameters prefixed with MASTER are used by
master nodes, and the ones prefixed with WORKER by worker nodes. When not specified, default configuration parameters
are used.

| Configuration parameter | Environment variable  | Default value |
| ----------------------- | --------------------- | ------------- |
| Master upstream host    | MASTER_UPSTREAM_HOST  | 127.0.0.1     |
| Master upstream port    | MASTER_UPSTREAM_PORT  | 50007         |
| Worker listen address   | WORKER_LISTEN_ADDRESS | 127.0.0.1     |
| Worker listen port      | WORKER_LISTEN_PORT    | 50007         |

## Collect statistics

On Jetson devices, it is possible to collect hardware usage statistics. This is implemented by using
[jetson_stats](https://github.com/rbonghi/jetson_stats) toolkit.

The Python requirements can be installed with the following command:
```
pip3 install -r requirements.txt
```

Once installed, statistics collection can be started by running:
```
make collect-stats
```

The collection program can be stopped with `Ctrl+C`.

## Uninstall

The locally stored assets can be removed by running the following command:
```
make clean
```
