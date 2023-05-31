#!/bin/bash

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
      deploy_resource "learning_worker_deployment"
      wait_for_deployment "learning-worker"
      if [ $SPARSE_DATASOURCE_USE_EXTERNAL_LINK == "yes" ]; then
        deploy_resource "learning_worker_nodeport"
      else
        deploy_resource "learning_worker_clusterip"
      fi

      deploy_resource "learning_datasource"
      ;;
    "edge_split")
      deploy_resource "learning_worker_deployment"
      wait_for_deployment "learning-worker"
      if [ $SPARSE_DATASOURCE_USE_EXTERNAL_LINK == "yes" ]; then
        deploy_resource "learning_worker_nodeport"
      else
        deploy_resource "learning_worker_clusterip"
      fi

      deploy_resource "learning_client"
      ;;
    "fog_offloading")
      deploy_resource "learning_worker_deployment"
      wait_for_deployment "learning-worker"
      if [ $SPARSE_DATASOURCE_USE_EXTERNAL_LINK == "yes" ]; then
        deploy_resource "learning_worker_nodeport"
      else
        deploy_resource "learning_worker_clusterip"
      fi

      deploy_resource "learning_intermediate"
      wait_for_deployment "learning-intermediate"
      deploy_resource "learning_datasource"
      ;;
    *)
      deploy_resource "learning_aio"
      ;;
  esac
}

deploy_nodes
