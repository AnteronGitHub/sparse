import asyncio

from sparse_framework.dl import ModelServeServer

from utils import parse_arguments, _get_benchmark_log_file_prefix

if __name__ == '__main__':
    args = parse_arguments()

    asyncio.run(ModelServeServer().start())
