#!/bin/bash

init_environment () {
  read -p "Name of the experiment suite (aio/edge_offloading/edge_split/fog_offloading): " SPARSE_SUITE
  export SPARSE_SUITE=${SPARSE_SUITE:-aio}

  read -p "Model to be used (default VGG): " SPARSE_MODEL
  export SPARSE_MODEL=${SPARSE_MODEL:-VGG}

  read -p "Dataset to be used (default CIFAR10): " SPARSE_DATASET
  export SPARSE_DATASET=${SPARSE_DATASET:-CIFAR10}

  read -p "Batch size to be used in training (default 64): " SPARSE_BATCH_SIZE
  export SPARSE_BATCH_SIZE=${SPARSE_BATCH_SIZE:-64}

  read -p "Number of batches to be used in training (default 64): " SPARSE_BATCHES
  export SPARSE_BATCHES=${SPARSE_BATCHES:-64}

  if [ $SPARSE_SUITE == "edge_split" ] || [ $SPARSE_SUITE == "fog_offloading" ]; then
    read -p "Use compression (default 1): " SPARSE_USE_COMPRESSION
    export SPARSE_USE_COMPRESSION=${SPARSE_USE_COMPRESSION:-1}
  else
    export SPARSE_USE_COMPRESSION=0
  fi

  if [ $SPARSE_USE_COMPRESSION == 1 ]; then
    read -p "Deprune props to be used in training (default 'budget:16;epochs:2;pruneState:1,budget:128;epochs:2;pruneState:1'): " SPARSE_DEPRUNE_PROPS
  else
    read -p "How many epochs to run training for (default '4'): " SPARSE_EPOCHS
  fi

  export SPARSE_DEPRUNE_PROPS=${SPARSE_DEPRUNE_PROPS:-budget:16;epochs:2;pruneState:1,budget:128;epochs:2;pruneState:1}
  export SPARSE_EPOCHS=${SPARSE_EPOCHS:-4}

  read -p "Network interface to monitor in benchmarks (default '' (all)): " SPARSE_MONITOR_NIC

  export SPARSE_MONITOR_NIC=${SPARSE_MONITOR_NIC:-""}

  if [ $SPARSE_SUITE == "fog_offloading" ]
  then
    export SPARSE_DATASOURCE_DOWNSTREAM_HOST="learning-intermediate"
    export SPARSE_DATASOURCE_DOWNSTREAM_PORT=50008
  else
    export SPARSE_DATASOURCE_DOWNSTREAM_HOST="learning-worker"
    export SPARSE_DATASOURCE_DOWNSTREAM_PORT=50007
  fi
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
    "edge_offloading")
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
