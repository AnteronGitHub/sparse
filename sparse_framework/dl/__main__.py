import asyncio

from .node import ParameterServer

if __name__ == "__main__":
    asyncio.run(ParameterServer().start())
