#!/bin/bash
# See: https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048

LIB_PATH=lib
mkdir -p ${LIB_PATH}

# Install library dependencies
sudo apt-get install -y libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev

# Clone torchvision sources
export BUILD_VERSION=v0.9.0
TORCHVISION_PATH=${LIB_PATH}/torchvision
git clone --branch ${BUILD_VERSION} https://github.com/pytorch/vision ${TORCHVISION_PATH}

CDIR=$PWD
cd ${TORCHVISION_PATH}
python3 setup.py install --user
cd ${CDIR}
