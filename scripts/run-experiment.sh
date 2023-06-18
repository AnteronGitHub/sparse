#!/bin/bash

create_sparse_namespace () {
  sudo kubectl create namespace sparse
}

deploy_resource () {
  cat k8s/$1.yaml | envsubst | sudo kubectl create -f -
}

wait_for_deployment () {
  echo "Waiting for deployment '$1' to be available..."
  sudo kubectl wait --namespace sparse --for=condition=Available deploy/$1
}

deploy_nodes () {
  deploy_resource "sparse_monitor"
  deploy_resource "model_server"

  case $SPARSE_SUITE in
    "edge_offloading")
      deploy_resource $SPARSE_APPLICATION"_worker_deployment"
      wait_for_deployment $SPARSE_APPLICATION"-worker"
      if [ $SPARSE_DATASOURCE_USE_EXTERNAL_LINK == "yes" ]; then
        deploy_resource $SPARSE_APPLICATION"_worker_nodeport"
      else
        deploy_resource $SPARSE_APPLICATION"_worker_clusterip"
      fi

      deploy_resource $SPARSE_APPLICATION"_datasource"
      ;;
    "edge_split")
      deploy_resource $SPARSE_APPLICATION"_worker_deployment"
      wait_for_deployment $SPARSE_APPLICATION"-worker"
      if [ $SPARSE_DATASOURCE_USE_EXTERNAL_LINK == "yes" ]; then
        deploy_resource $SPARSE_APPLICATION"_worker_nodeport"
      else
        deploy_resource $SPARSE_APPLICATION"_worker_clusterip"
      fi

      deploy_resource $SPARSE_APPLICATION"_client"
      ;;
    "fog_offloading")
      deploy_resource $SPARSE_APPLICATION"_worker_deployment"
      wait_for_deployment $SPARSE_APPLICATION"-worker"
      if [ $SPARSE_DATASOURCE_USE_EXTERNAL_LINK == "yes" ]; then
        deploy_resource $SPARSE_APPLICATION"_worker_nodeport"
      else
        deploy_resource $SPARSE_APPLICATION"_worker_clusterip"
      fi

      deploy_resource $SPARSE_APPLICATION"_intermediate"
      wait_for_deployment $SPARSE_APPLICATION"-intermediate"
      deploy_resource $SPARSE_APPLICATION"_datasource"
      ;;
    *)
      deploy_resource $SPARSE_APPLICATION"_aio"
      ;;
  esac
}

create_sparse_namespace
deploy_nodes
