# Sparse

This repository contains source code for Stream Processing Architecture for Resource Subtle Environments (or just
Sparse for short). Additionally, sample applications utilizing Sparse for deep learning can be found in examples
directory.

## Compatibility

The software has been tested, and thus can be considered compatible with, the following devices and the following
software:

| Device            | JetPack version | Python version | PyTorch version | Docker version | Base image                                     | Docker tag suffix |
| ----------------- | --------------- | -------------- | --------------- | -------------- | ---------------------------------------------- | ------------------ |
| Jetson AGX Xavier | 5.0 preview     | 3.8.10         | 1.12.0a0        | 20.10.12       | nvcr.io/nvidia/l4t-pytorch:r34.1.0-pth1.12-py3 | jp50               |
| Jetson Nano 2GB   | 4.6.1           | 3.6.9          | 1.9.0           | 20.10.7        | nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.9-py3  | jp461              |
| Lenovo ThinkPad   | -               | 3.8.12         | 1.11.0          | 20.10.15       | pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime | amd64              |

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

#### Base image

Software dependencies are included in Docker images which can be built locally. Run the following command in order to
build the base image for Deep Learning applications using the stream processing framework:
```
docker build . -t sparse:amd64
```

When building image for Jetson devices, specify the appropriate base image with build argument. See compatibility table
above for the appropriate images, as well as tags that one should use. For example, in order to build the base image
for JetPack 5.0, one should run the following command:

```
docker build . --build-arg BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r34.1.0-pth1.12-py3 -t sparse:jp50
```

#### Application image

Use the previously built base image as the base image for applications using the framework. For instance, in order to
build image for split learning client, using the example provided in the repository, run:

```
cd examples/split_learning
docker build . -f Dockerfile.client -t split_learning:client.amd64
```

## Run program

### Python

With dependencies installed locally, agents can be started with the following command, by replacing \<node\> with the
corresponding node (client/server):

```
python3 src/split_training_<node>.py
```

### Docker

Use the previously built docker files to create new application containers. For instance, run the following command to
start a split learning server on amd64 supported devices:

```
docker run --rm -it split_learning:server.amd64
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
