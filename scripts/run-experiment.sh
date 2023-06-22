#!/bin/bash

create_sparse_namespace () {
  sudo kubectl create namespace sparse
}

deploy_sparse_resource () {
  cat k8s/$1.yaml | envsubst | sudo kubectl create -f -
}

deploy_resource () {
  cat examples/$1/k8s/$2.yaml | envsubst | sudo kubectl create -f -
}

wait_for_deployment () {
  echo "Waiting for deployment '$1' to be available..."
  sudo kubectl wait --namespace sparse --for=condition=Available deploy/$1
}

deploy_nodes () {
  deploy_sparse_resource "sparse_monitor"
  deploy_sparse_resource "model_server"

  case $SPARSE_SUITE in
    "edge_offloading")
      deploy_resource $SPARSE_EXAMPLE "worker_deployment"
      wait_for_deployment $SPARSE_EXAMPLE"-worker"
      if [ $SPARSE_DATASOURCE_USE_EXTERNAL_LINK == "yes" ]; then
        deploy_resource $SPARSE_EXAMPLE "worker_nodeport"
      else
        deploy_resource $SPARSE_EXAMPLE "worker_clusterip"
      fi

      deploy_resource $SPARSE_EXAMPLE "datasource"
      ;;
    "edge_split")
      deploy_resource $SPARSE_EXAMPLE "worker_deployment"
      wait_for_deployment $SPARSE_EXAMPLE"-worker"
      if [ $SPARSE_DATASOURCE_USE_EXTERNAL_LINK == "yes" ]; then
        deploy_resource $SPARSE_EXAMPLE "worker_nodeport"
      else
        deploy_resource $SPARSE_EXAMPLE "worker_clusterip"
      fi

      deploy_resource $SPARSE_EXAMPLE "client"
      ;;
    "fog_offloading")
      deploy_resource $SPARSE_EXAMPLE "worker_deployment"
      wait_for_deployment $SPARSE_EXAMPLE"-worker"
      if [ $SPARSE_DATASOURCE_USE_EXTERNAL_LINK == "yes" ]; then
        deploy_resource $SPARSE_EXAMPLE "worker_nodeport"
      else
        deploy_resource $SPARSE_EXAMPLE "worker_clusterip"
      fi

      deploy_resource $SPARSE_EXAMPLE "intermediate"
      wait_for_deployment $SPARSE_EXAMPLE"-intermediate"
      deploy_resource $SPARSE_EXAMPLE "datasource"
      ;;
    *)
      deploy_resource $SPARSE_EXAMPLE "aio"
      ;;
  esac
}

create_sparse_namespace
deploy_nodes
