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

To ease development, the resource templates map source code from host path `/opt/sparse/src` to the containers. If the
repository was not cloned to `/opt` directory, you can create a symbolic link to your repository:
```
sudo ln -s <sparse_repo_path> /opt/sparse
```

### Label cluster nodes

The templates place pods to hosts based on node labels. To enable all pods to be placed on a node (e.g. in development
environments), you can use label aio:

```
kubectl label node <dev-node> sparse/node=aio
```

If you want to separate data sources and worker nodes, set the following labels to the appropriate cluster nodes:
```
kubectl label node <source-node> sparse/node=datasource
kubectl label node <worker-node> sparse/node=worker
```

## Resource creation

Once the cluster has been initialized, create a namespace for sparse:
```
kubectl create namespace sparse
```

Then, the sparse pipeline resources can be created as follows (with monitoring included):
```
kubectl create -f k8s/sparse_monitor.yaml
kubectl create -f k8s/learning_unsplit_offloaded.yaml
```
