# Sparse resources in Kubernetes

This directory contains resource templates for deploying sparse in a Kubernetes cluster.

*The templates are still under development and do not meet typical production deployment standards!*

## Prerequisites
The templates use direct host paths for persistent volumes, and they assume that each docker image has been built
locally. Finally, resource placement is driven by manually set node labels.

Complete the following steps on each cluster node to prepare the deployment.

### Build docker images
Build all the docker images locally on each host, by running the following commands:
```
git clone https://github.com/AnteronGitHub/sparse.git -b k8s
cd sparse
make docker
```

### Set up host paths

Ensure that each cluster node has the following mount paths available for k8s host path volumes:
```
sudo mkdir /mnt/sparse/data
sudo mkdir /mnt/sparse/run
sudo mkdir /mnt/sparse/stats
```

### Label cluster nodes

The templates place pods to hosts based on node labels. Set the following labels on nodes that you want to use as data
sources and workers:
```
kubectl label node <source-node> sparse/node=datasource
kubectl label node <worker-node> sparse/node=worker
```

## Resource creation

Once the cluster has been initialized, the resources can be created as follows:
```
kubectl create -f k8s/sparse_monitor.yaml
kubectl create -f k8s/learning_unsplit_offloaded.yaml
```
