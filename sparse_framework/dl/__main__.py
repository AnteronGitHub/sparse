import asyncio

from .node import ModelServer

if __name__ == "__main__":
    asyncio.run(ModelServer().start())
