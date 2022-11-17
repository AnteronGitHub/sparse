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
make run-learning-unsplit
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

To test that the program was installed correctly, run the inference suite with an unsplit model with the following
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


### Statistics

In order to collect benchmark statistics for training or inference, before running a suite with the above instructions,
first start the monitor server by running the following command:

```
make run-sparse-monitor
```

## Configuration

Nodes can be configured with environment variables. Environment variables can be specified inline, or with a dotenv
file in the data directory.

When using the Make scripts for running the software, the dotenv file should be `./data/.env`:
```
mkdir data
touch data/.env
```

### Configuration Options

Parameters prefixed with MASTER are used by master nodes, and the ones prefixed with WORKER by worker nodes. When not
specified, default configuration parameters are used.

| Configuration parameter | Environment variable  | Default value |
| ----------------------- | --------------------- | ------------- |
| Master upstream host    | MASTER_UPSTREAM_HOST  | 127.0.0.1     |
| Master upstream port    | MASTER_UPSTREAM_PORT  | 50007         |
| Worker listen address   | WORKER_LISTEN_ADDRESS | 127.0.0.1     |
| Worker listen port      | WORKER_LISTEN_PORT    | 50007         |

By convention, the port 50007 will be used for workers that expect raw data, i.e. unsplit workers, and the first
splits. The port 50008 is used by workers that expect the first task split output data, i.e. final split nodes. While
this port mapping is not a technical requirement, the Make scripts follow it.

## Multi-node deployment

In order to set up a pipeline on multiple hosts, make sure that the master nodes have IP connectivity to the worker
nodes. Then, for each master node, specify the IP address of the worker that the task will be offloaded to. If using
the Make scripts to start nodes, this is the only configuration required.

### Example: Three node split training
This is an example on how to configure split training across three nodes: a data source, an intermediate worker, and a
final worker. The data source will send the feature vectors to the intermediate worker, which will process the first
split of the task. The intermediate node will then send the results of the first split to the final worker which will
run the final split to finish the task.

Assume that the nodes have the following IP addressing in place:

| Node                  | IP address    |
| --------------------- | ------------- |
| Data source           | 10.49.2.1     |
| Intermediate worker   | 10.49.2.2     |
| Final worker          | 10.49.2.3     |

1. Start the final worker node

Run the following command to start the final training split in the *final worker* node:
```
make run-learning-split-final
```

2. Configure and start the intermediate worker

Add a .env file with the following contents in the *intermediate worker* node:
```
MASTER_UPSTREAM_HOST=10.49.2.3
```

Then start the intermediate worker by running the following command:
```
make run-learning-split-intermediate
```

3. Configure and start the data source

Add a .env file with the following contents in the *data source* node:
```
MASTER_UPSTREAM_HOST=10.49.2.2
```

Then start the data source by running the following command:
```
make run-learning-data-source
```

## Uninstall

The locally stored assets can be removed by running the following command:
```
make clean
```
