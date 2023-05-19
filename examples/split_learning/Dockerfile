# | SDK            | Arch tag |
# | -------------- | -------- |
# | JetPack 5.0    | jp50     |
# | JetPack 4.6.1  | jp461    |
ARG ARCH_TAG=amd64
FROM sparse/pytorch

WORKDIR /app

COPY . .

CMD ["python3", "learning_aio.py"]
