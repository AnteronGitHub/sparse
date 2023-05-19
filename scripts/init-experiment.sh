#!/bin/bash

init_environment () {
  read -p "Name of the experiment suite (aio/edge_offloading/cloud_offloading/edge_split/fog_offloading): " SPARSE_SUITE
  read -p "Model to be used (default VGG): " SPARSE_MODEL
  read -p "Dataset to be used (default CIFAR10): " SPARSE_DATASET
  read -p "Batch size to be used in training (default 64): " SPARSE_BATCH_SIZE
  read -p "Number of batches to be used in training (default 64): " SPARSE_BATCHES

  SPARSE_SUITE=${SPARSE_SUITE:-aio}
  SPARSE_MODEL=${SPARSE_MODEL:-VGG}
  SPARSE_DATASET=${SPARSE_DATASET:-CIFAR10}
  SPARSE_BATCH_SIZE=${SPARSE_BATCH_SIZE:-64}
  SPARSE_BATCHES=${SPARSE_BATCHES:-64}

  if [ $SPARSE_SUITE == "fog_offloading" ]
  then
    export SPARSE_DATASOURCE_DOWNSTREAM_HOST="learning-intermediate"
    export SPARSE_DATASOURCE_DOWNSTREAM_PORT=50008
  else
    export SPARSE_DATASOURCE_DOWNSTREAM_HOST="learning-worker"
    export SPARSE_DATASOURCE_DOWNSTREAM_PORT=50007
  fi

  export SPARSE_SUITE
  export SPARSE_MODEL
  export SPARSE_DATASET
  export SPARSE_BATCH_SIZE
  export SPARSE_BATCHES
}

create_sparse_namespace () {
  sudo kubectl create namespace sparse
}

deploy_node () {
  cat k8s/$1.yaml | envsubst | sudo kubectl create -f -
}

wait_for_deployment () {
  echo "Waiting for deployment '$1' to be available..."
  sudo kubectl wait --namespace sparse --for=condition=Available deploy/$1
}

deploy_nodes () {
  deploy_node "sparse_monitor"

  case $SPARSE_SUITE in
    "edge_offloading" | "cloud_offloading")
      deploy_node "learning_worker"
      wait_for_deployment "learning-worker"
      deploy_node "learning_datasource"
      ;;
    "edge_split")
      deploy_node "learning_worker"
      wait_for_deployment "learning-worker"
      deploy_node "learning_client"
      ;;
    "fog_offloading")
      deploy_node "learning_worker"
      wait_for_deployment "learning-worker"
      deploy_node "learning_intermediate"
      wait_for_deployment "learning-intermediate"
      deploy_node "learning_datasource"
      ;;
    *)
      deploy_node "learning_aio"
      ;;
  esac
}

init_environment
create_sparse_namespace
deploy_nodes
