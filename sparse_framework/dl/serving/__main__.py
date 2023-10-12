import asyncio

from sparse_framework import TCPServer

if __name__ == "__main__":
    from . import ModelServer, DiskModelRepository

    server = TCPServer(listen_address = '0.0.0.0', listen_port = 50006)
    asyncio.run(server.serve(lambda: ModelServer(model_repository=DiskModelRepository())))
