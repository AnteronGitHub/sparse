import asyncio

from sparse_framework.dl import InferenceServer

from utils import parse_arguments

if __name__ == '__main__':
    args = parse_arguments()

    asyncio.run(InferenceServer().start())
