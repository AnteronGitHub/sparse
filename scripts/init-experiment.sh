#!/bin/bash

init_environment () {

  # Experiment specs
  read -p "Which example to run (splitnn/deprune)? " SPARSE_EXAMPLE
  export SPARSE_EXAMPLE=${SPARSE_EXAMPLE:-splitnn}

  read -p "Run learning or inference (default learning)? " SPARSE_APPLICATION
  export SPARSE_APPLICATION=${SPARSE_APPLICATION:-learning}

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

  read -p "How many data sources to run (default 1): " SPARSE_NO_DATASOURCES
  export SPARSE_NO_DATASOURCES=${SPARSE_NO_DATASOURCES:-1}

  read -p "Specify the data source cpu limitation (default 400m): " SPARSE_DATASOURCE_CPU_LIMIT
  export SPARSE_DATASOURCE_CPU_LIMIT=${SPARSE_DATASOURCE_CPU_LIMIT:-400m}

  if [ $SPARSE_EXAMPLE == "deprune" ]; then
    read -p "Deprune props to be used in training (default 'budget:16;epochs:2;pruneState:1,budget:128;epochs:2;pruneState:1'): " SPARSE_DEPRUNE_PROPS
    read -p "Feature compression factor (default '1'): " SPARSE_FEATURE_COMPRESSION_FACTOR
    read -p "Resolution compression factor (default '1'): " SPARSE_RESOLUTION_COMPRESSION_FACTOR
  else
    read -p "How many epochs to run training for (default '4'): " SPARSE_EPOCHS
  fi

  export SPARSE_DEPRUNE_PROPS=${SPARSE_DEPRUNE_PROPS:-budget:16;epochs:2;pruneState:1,budget:128;epochs:2;pruneState:1}
  export SPARSE_FEATURE_COMPRESSION_FACTOR=${SPARSE_FEATURE_COMPRESSION_FACTOR:-1}
  export SPARSE_RESOLUTION_COMPRESSION_FACTOR=${SPARSE_RESOLUTION_COMPRESSION_FACTOR:-1}
  export SPARSE_EPOCHS=${SPARSE_EPOCHS:-4}


  # Monitoring specs
  read -p "Network interface to monitor in benchmarks (default '' (all)): " SPARSE_MONITOR_NIC

  export SPARSE_MONITOR_NIC=$SPARSE_MONITOR_NIC


  # Deployment specs
  read -p "Use external link for data source (default 'no'): " SPARSE_DATASOURCE_USE_EXTERNAL_LINK
  export SPARSE_DATASOURCE_USE_EXTERNAL_LINK=${SPARSE_DATASOURCE_USE_EXTERNAL_LINK:-"no"}
  if [ $SPARSE_DATASOURCE_USE_EXTERNAL_LINK == "yes" ]; then
    read -p "External IP for downstream link: " SPARSE_DATASOURCE_DOWNSTREAM_HOST
    read -p "Port for downstream link (default 30007): " SPARSE_DATASOURCE_DOWNSTREAM_PORT

    export SPARSE_DATASOURCE_DOWNSTREAM_HOST=$SPARSE_DATASOURCE_DOWNSTREAM_HOST
    export SPARSE_DATASOURCE_DOWNSTREAM_PORT=${SPARSE_DATASOURCE_DOWNSTREAM_PORT:-"30007"}
  elif [ $SPARSE_SUITE == "fog_offloading" ]; then
    export SPARSE_DATASOURCE_DOWNSTREAM_HOST=$SPARSE_EXAMPLE"-intermediate"
    export SPARSE_DATASOURCE_DOWNSTREAM_PORT=50008
  else
    export SPARSE_DATASOURCE_DOWNSTREAM_HOST=$SPARSE_EXAMPLE"-worker"
    export SPARSE_DATASOURCE_DOWNSTREAM_PORT=50007
  fi
}

init_environment
