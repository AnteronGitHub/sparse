# Edge Deep Learning

This repository contains source code for experiments about Deep learning in the Edge.

## Compatibility

The software has been tested, and thus can be considered compatible with, the following devices and the following
software:

| Device            | JetPack version | Python version | Pytorch version | Docker version |
| ----------------- | --------------- | -------------- | --------------- | -------------- |
| Jetson AGX Xavier | 5.0 preview     | 3.8.10         | 1.12.0a0        | 20.10.12       |
| Jetson Nano 2GB   | 4.6.1           | 3.6.9          | 1.9.0           | 20.10.7        |
| Lenovo ThinkPad   | -               | 3.8.12         | 1.11.0          | 20.10.15       |

## Install

The repository uses PyTorch as the primary Deep Learning framework.
[`jetson-infrence`](https://github.com/dusty-nv/jetson-inference) repository, provided openly by NVIDIA, is also used
for some test suites on Jetson devices (see below).

### Jetson Nano 2GB
In order to use PyTorch on Jetson Nano 2GB, it has to be built using distributions provided by NVIDIA. The repository
includes installation scripts for doing this. Install dependencies by running the following command at the repository
root:
```
make jetson-dependencies
```

### Laptop

Unless otherwise specified, the experiment suites can be run on common laptops using CPU. For this, Python dependencies
need to be installed:

```
pip3 install -r requirements-client.txt
```

## Run program

### Server

In order to run split training over the network, before running the client program, start a server in a separate
process, by running the following command:
```
python3 src/server.py
```

The same command can also be invoked with make utility, by running:
```
make run-server
```

### Client

With software dependencies installed, the program can be run with make utility, by running the following command at the
repository root:
```
python3 src/client.py
```

```
make run
```

Alternatively, the dependencies can be installed into a Python virtual environment, by running the following command
(not recommended on resource-constrained devices, such as Jetson Nano):
```
make run-venv
```

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
