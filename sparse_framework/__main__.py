import asyncio

from .node import SparseNode

if __name__ == '__main__':
    asyncio.run(SparseNode().start())
