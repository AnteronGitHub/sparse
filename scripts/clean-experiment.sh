#!/bin/bash

delete_pod () {
  sudo kubectl delete --ignore-not-found -f k8s/$1.yaml
}

delete_pods () {
  case $SPARSE_SUITE in
    "edge_offloading")
      delete_pod "learning_datasource"
      delete_pod "learning_worker"
      ;;
    "edge_split")
      delete_pod "learning_client"
      delete_pod "learning_worker"
      ;;
    "fog_offloading")
      delete_pod "learning_datasource"
      delete_pod "learning_intermediate"
      delete_pod "learning_worker"
      ;;
    *)
      delete_pod "learning_aio"
      ;;
  esac
}

delete_pods
