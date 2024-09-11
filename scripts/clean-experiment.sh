#!/bin/bash

delete_sparse_resource () {
  sudo kubectl delete --ignore-not-found -f k8s/$1.yaml
}

delete_resource () {
  sudo kubectl delete --ignore-not-found -f examples/$1/k8s/$2.yaml
}

delete_pods () {
  SPARSE_EXAMPLE="splitnn"
  case $SPARSE_SUITE in
    "edge_offloading")
      delete_resource $SPARSE_EXAMPLE "datasource"
      delete_resource $SPARSE_EXAMPLE "worker_deployment"
      if [ $SPARSE_DATASOURCE_USE_EXTERNAL_LINK == "yes" ]; then
        delete_resource $SPARSE_EXAMPLE "worker_nodeport"
      else
        delete_resource $SPARSE_EXAMPLE "worker_clusterip"
      fi
      ;;
    "edge_split")
      delete_resource $SPARSE_EXAMPLE "client"
      delete_resource $SPARSE_EXAMPLE "worker_deployment"
      if [ $SPARSE_DATASOURCE_USE_EXTERNAL_LINK == "yes" ]; then
        delete_resource $SPARSE_EXAMPLE "worker_nodeport"
      else
        delete_resource $SPARSE_EXAMPLE "worker_clusterip"
      fi
      ;;
    "fog_offloading")
      delete_resource $SPARSE_EXAMPLE "datasource"
      delete_resource $SPARSE_EXAMPLE "intermediate"
      delete_resource $SPARSE_EXAMPLE "worker_deployment"
      if [ $SPARSE_DATASOURCE_USE_EXTERNAL_LINK == "yes" ]; then
        delete_resource $SPARSE_EXAMPLE "worker_nodeport"
      else
        delete_resource $SPARSE_EXAMPLE "worker_clusterip"
      fi
      ;;
    *)
      delete_resource $SPARSE_EXAMPLE "aio"
      ;;
  esac

#  delete_sparse_resource "model_server"
}

delete_pods
