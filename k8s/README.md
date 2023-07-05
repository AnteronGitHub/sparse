# Sparse resources in Kubernetes

This directory contains resource templates for deploying sparse in a Kubernetes cluster.

*The templates are still under development and do not meet typical production deployment standards!*

## Prerequisites
The templates use direct host paths for persistent volumes, and they assume that each docker image has been built
locally. Finally, resource placement is driven by manually set node labels.

*As a starting point, you should have a self-deployed Kubernetes cluster instance (we mainly use
[k3s](https://k3s.io/) for development)*. Once you have k8s installed, complete the following steps on each cluster
node to prepare the deployment.

### Build docker images
Build all the docker images locally on each host, by running the following commands:
```
git clone https://github.com/AnteronGitHub/sparse.git
cd sparse
make docker
```

### Set up host paths

To ease development, the resource templates map source code from host path `/opt/sparse/sparse_framework` to the containers. If the
repository was not cloned to `/opt` directory, you can create a symbolic link to your repository:
```
sudo ln -s <sparse_repo_path> /opt/sparse
```

### Label cluster nodes

The templates place pods to hosts based on node labels.

Firstly, one of the cluster nodes needs to be labeled as a model server. To do so, use the command below (with
<model-server> replaced as the name of the node):

```
kubectl label node <model-server> sparse/model-server=true
```

To enable all pods to be placed on a node (e.g. in development environments), you can use label aio:

```
kubectl label node <dev-node> sparse/node=aio
```

If you want to separate data sources and worker nodes, set the following labels to the appropriate cluster nodes:
```
kubectl label node <source-node> sparse/node=datasource
kubectl label node <worker-node> sparse/node=worker
```

## Setup and run experiments with the example applications

Once the cluster has been initialized, run the following command to set the environment variables needed for the
example applications interactively:
```
source scripts/init-experiment.sh
```

Then run an experiment with the following command:
```
make run-experiment
```

Once all of the data source pods are completed, the experiment resources can be removed with the following command:
```
make clean-experiment
```

The k8s stacks deploy statistics server for collecting benchmark measurements. By default, the statistics are collected
to csv files located in the `/var/lib/sparse/stats/` in the cluster nodes.

The locally stored assets can be removed with the following command:
```
make clean
```
