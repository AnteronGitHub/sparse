#!/bin/bash

mkdir -p $1

echo "Fetching test data"
wget https://github.com/dusty-nv/jetson-inference/raw/master/data/images/$2 -O $1/$2
