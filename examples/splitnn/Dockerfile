# | SDK            | Arch tag |
# | -------------- | -------- |
# | JetPack 5.0    | jp50     |
# | JetPack 4.6.1  | jp461    |
ARG BASE_IMAGE=pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
FROM $BASE_IMAGE

WORKDIR /app

COPY sparse_framework-1.2.0-py3-none-any.whl sparse_framework-1.2.0-py3-none-any.whl
RUN pip3 install sparse_framework-1.2.0-py3-none-any.whl sparse_framework-1.2.0-py3-none-any.whl

COPY . .

CMD ["python3", "worker.py"]
