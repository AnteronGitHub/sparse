#!/bin/bash

delete_resource () {
  sudo kubectl delete --ignore-not-found -f k8s/$1.yaml
}

delete_pods () {
  case $SPARSE_SUITE in
    "edge_offloading")
      delete_resource $SPARSE_APPLICATION"_datasource"
      delete_resource $SPARSE_APPLICATION"_worker_deployment"
      if [ $SPARSE_DATASOURCE_USE_EXTERNAL_LINK == "yes" ]; then
        delete_resource $SPARSE_APPLICATION"_worker_nodeport"
      else
        delete_resource $SPARSE_APPLICATION"_worker_clusterip"
      fi
      ;;
    "edge_split")
      delete_resource $SPARSE_APPLICATION"_client"
      delete_resource $SPARSE_APPLICATION"_worker_deployment"
      if [ $SPARSE_DATASOURCE_USE_EXTERNAL_LINK == "yes" ]; then
        delete_resource $SPARSE_APPLICATION"_worker_nodeport"
      else
        delete_resource $SPARSE_APPLICATION"_worker_clusterip"
      fi
      ;;
    "fog_offloading")
      delete_resource $SPARSE_APPLICATION"_datasource"
      delete_resource $SPARSE_APPLICATION"_intermediate"
      delete_resource $SPARSE_APPLICATION"_worker_deployment"
      if [ $SPARSE_DATASOURCE_USE_EXTERNAL_LINK == "yes" ]; then
        delete_resource $SPARSE_APPLICATION"_worker_nodeport"
      else
        delete_resource $SPARSE_APPLICATION"_worker_clusterip"
      fi
      ;;
    *)
      delete_resource $SPARSE_APPLICATION"_aio"
      ;;
  esac

  delete_resource "model_server"
  delete_resource "sparse_monitor"
}

delete_pods
