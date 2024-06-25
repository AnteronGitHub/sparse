#!/bin/bash

sudo kubectl apply -f k8s/worker_deployment.yaml
sudo kubectl apply -f k8s/worker_clusterip.yaml
