#!/bin/bash

mkdir -p $1

echo "Fetching test data"
wget https://github.com/dusty-nv/jetson-inference/raw/master/data/images/orange_0.jpg -O $1/$2
