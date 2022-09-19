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
| Lenovo ThinkPad   | -               | 3.8.12         | 1.11.0          | 20.10.15       | pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime | amd64              |

## Install

The repository uses PyTorch as the primary Deep Learning framework. Software dependencies can be installed with pip or
by using Docker.

### Make

Software can be installed with make utility, by running the following command:
```
make all
```

## Run training

### All-In-One

To test that the program was installed correctly, run the training suite with an unsplit model with the following
command:

```
make run-learning-aio
```

### Unsplit offloaded

First start the unsplit training server with the following command:
```
make run-learning-aio
```

Then start the data source with the following command:
```
make run-learning-data-source
```

### Split offloaded

First start the split training nodes with the following command:
```
make run-learning-split
```

Then start the data source with the following command:
```
make run-learning-data-source
```

## Run Inference

### All-In-One

To test that the program was installed correctly, run the infernce suite with an unsplit model with the following
command:

```
make run-inference-aio
```

### Unsplit offloaded

First start the unsplit inference server with the following command:
```
make run-inference-unsplit
```

Then start the data source with the following command:
```
make run-inference-data-source
```

### Split offloaded

First start the split inference nodes with the following command:
```
make run-inference-split
```


## Collect statistics

In order to collect benchmark statistics for training, before running the suite with the above instructions, start the
monitor server by running the following command:

```
make run-learning-monitor
```

The statistics for inference can be collected with the following command:
```
make run-inference-monitor
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

## Uninstall

The locally stored assets can be removed by running the following command:
```
make clean
```
