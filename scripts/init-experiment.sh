#!/bin/bash

init_environment () {
  SPARSE_EXAMPLE="splitnn"

  # Experiment specs
  read -p "Name of the experiment suite (edge_offloading/aio/edge_split/fog_offloading): " SPARSE_SUITE
  export SPARSE_SUITE=${SPARSE_SUITE:-edge_offloading}

  read -p "Model to be used (default VGG): " SPARSE_MODEL
  export SPARSE_MODEL=${SPARSE_MODEL:-VGG}

  read -p "Dataset to be used (default CIFAR10): " SPARSE_DATASET
  export SPARSE_DATASET=${SPARSE_DATASET:-CIFAR10}

  read -p "Number of samples per dataset (default 64): " SPARSE_NO_SAMPLES
  export SPARSE_NO_SAMPLES=${SPARSE_NO_SAMPLES:-64}

  read -p "How many data sources to run (default 1): " SPARSE_NO_DATASOURCES
  export SPARSE_NO_DATASOURCES=${SPARSE_NO_DATASOURCES:-1}

  read -p "How many models to serve (default 1): " SPARSE_NO_MODELS
  export SPARSE_NO_MODELS=${SPARSE_NO_MODELS:-1}

  read -p "Use scheduling (default 1): " SPARSE_USE_SCHEDULING
  export SPARSE_USE_SCHEDULING=${SPARSE_USE_SCHEDULING:-1}

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
