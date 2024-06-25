#!/bin/bash

sudo kubectl delete -f k8s/worker_deployment.yaml
sudo kubectl delete -f k8s/worker_clusterip.yaml
