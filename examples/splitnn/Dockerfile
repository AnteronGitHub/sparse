# | SDK            | Arch tag |
# | -------------- | -------- |
# | JetPack 5.0    | jp50     |
# | JetPack 4.6.1  | jp461    |
ARG TAGNAME=pytorch
FROM anterondocker/sparse-framework:$TAGNAME

WORKDIR /app

COPY . .

CMD ["python3", "aio.py"]
