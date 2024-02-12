# Specify base image for other devices with build argument.
#
# See below which images to use:
# | SDK            | Base image                                       |
# | -------------- | ------------------------------------------------ |
# | JetPack 5.0    | nvcr.io/nvidia/l4t-pytorch:r34.1.0-pth1.12-py3   |
# | JetPack 4.6.1  | nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.9-py3    |

FROM python:3.12-slim

ARG INSTALL_LOCATION=/usr/lib
ARG PY_REQUIREMENTS=requirements.txt

RUN mkdir -p $INSTALL_LOCATION

ENV PYTHONPATH=$PYTHONPATH:$INSTALL_LOCATION

COPY $PY_REQUIREMENTS requirements.txt
RUN pip3 install -r requirements.txt

WORKDIR $INSTALL_LOCATION/sparse_framework
COPY sparse_framework $INSTALL_LOCATION/sparse_framework

CMD ["python3", "-m", "sparse_framework.stats"]
