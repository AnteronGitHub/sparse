import asyncio

from .monitor_server import MonitorServer

if __name__ == '__main__':
    asyncio.run(MonitorServer().start())
