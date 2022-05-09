# Specify base image for other devices with build argument.
#
# See below which images to use:
# | SDK            | Base image                                       |
# | -------------- | ------------------------------------------------ |
# | JetPack 5.0    | nvcr.io/nvidia/l4t-pytorch:r34.1.0-pth1.12-py3   |
# | JetPack 4.6.1  | nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.9-py3    |

ARG BASE_IMAGE=pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

FROM $BASE_IMAGE

WORKDIR /app
COPY src sparse
