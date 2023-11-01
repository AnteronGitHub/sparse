from .monitor_container import MonitorContainer
from .network_monitor import NetworkMonitor
from .time_monitor import TimeMonitor
from .training_monitor import TrainingMonitor

class NodeMonitor(MonitorContainer):
    def __init__(self, nic):
        super().__init__([TimeMonitor(), NetworkMonitor(nic), TrainingMonitor()])

    def batch_processed(self, batch_size, loss):
        for monitor in self.monitors:
            if type(monitor).__name__ == 'TrainingMonitor':
                monitor.add_point(newly_processed_samples = batch_size, loss = loss)

    def task_processed(self):
        for monitor in self.monitors:
            if type(monitor).__name__ == 'TrainingMonitor':
                monitor.add_point(newly_processed_tasks = 1)

    def connection_timeout(self):
        for monitor in self.monitors:
            if type(monitor).__name__ == 'NetworkMonitor':
                monitor.add_connection_timeout()

    def broken_pipe_error(self):
        for monitor in self.monitors:
            if type(monitor).__name__ == 'NetworkMonitor':
                monitor.add_broken_pipe_error()

    def receive_message(self, payload : dict):
        if payload['event'] == 'batch_processed':
            self.batch_processed(payload['batch_size'], payload['loss'])
        elif payload['event'] == 'task_processed':
            self.task_processed()
        elif payload['event'] == 'connection_timeout':
            self.connection_timeout()
        elif payload['event'] == 'broken_pipe_error':
            self.broken_pipe_error()
