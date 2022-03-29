#!/bin/bash
# See: https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048

# Install prerequisites
sudo apt-get install -y python3-pip libopenblas-base libopenmpi-dev 

# Download wheel provided by NVIDIA
WHEEL_PATH=build/wheels
mkdir -p ${WHEEL_PATH}

PYTORCH_VERSION=1.8.0    # This version matches with the NVIDIA url below
WHEEL_FILE=${WHEEL_PATH}/torch-${PYTORCH_VERSION}-cp36-cp36m-linux_aarch64.whl
wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O ${WHEEL_FILE}

# Install Python requirements
PIP=pip3
$PIP install --force-reinstall Cython numpy==1.19.4 ${WHEEL_FILE}
