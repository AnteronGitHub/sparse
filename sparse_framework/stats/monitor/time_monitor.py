from time import time

from .base_monitor import BaseMonitor

class TimeMonitor(BaseMonitor):
    def __init__(self):
        self.initial_time = None

    def get_metrics(self):
        return ['timestamp']

    def get_stats(self):
        if self.initial_time is None:
            self.initial_time = time()

        return [time() - self.initial_time]

