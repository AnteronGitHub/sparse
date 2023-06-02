#!/bin/bash

delete_resource () {
  sudo kubectl delete --ignore-not-found -f k8s/$1.yaml
}

delete_pods () {
  case $SPARSE_SUITE in
    "edge_offloading")
      delete_resource "learning_datasource"
      delete_resource "learning_worker_deployment"
      delete_resource "learning_worker_nodeport"
      ;;
    "edge_split")
      delete_resource "learning_client"
      delete_resource "learning_worker_deployment"
      delete_resource "learning_worker_nodeport"
      ;;
    "fog_offloading")
      delete_resource "learning_datasource"
      delete_resource "learning_intermediate"
      delete_resource "learning_worker_deployment"
      delete_resource "learning_worker_nodeport"
      ;;
    *)
      delete_resource "learning_aio"
      ;;
  esac

  delete_resource "model_server"
  delete_resource "sparse_monitor"
}

delete_pods
