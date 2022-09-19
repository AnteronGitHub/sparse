# | SDK            | Arch tag |
# | -------------- | -------- |
# | JetPack 5.0    | jp50     |
# | JetPack 4.6.1  | jp461    |
FROM sparse/pytorch

RUN pip install matplotlib

WORKDIR /app

COPY . .

CMD ["python3", "benchmark.py"]
