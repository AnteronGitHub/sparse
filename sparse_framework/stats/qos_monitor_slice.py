import asyncio

from ..node import SparseSlice

from .monitor_daemon import MonitorDaemon

class SparseQoSMonitorSlice(SparseSlice):
    """Quality of Service Monitor Slice maintains a coroutine for monitoring the runtime performance of the node.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats_queue = None

    def get_futures(self, futures):
        self.stats_queue = asyncio.Queue()
        monitor_daemon = MonitorDaemon(self.stats_queue)

        futures.append(monitor_daemon.start())
        return futures

