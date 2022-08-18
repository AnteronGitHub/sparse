from time import time

class Monitor():
    def __init__(self):
        self.initial_time = None

    def get_metrics(self):
        return 'timestamp'

    def start(self):
        self.initial_time = time()

    def read_stats(self):
        if self.initial_time is None:
            self.initial_time = time()

        return time() - self.initial_time

    def log_stats(self):
        file_logger = FileLogger()
        file_logger.log_row(self.get_metrics())
        while True:
            try:
                file_logger.log_row(self.read_stats())
                time.sleep(1)
            except KeyboardInterrupt:
                print("Stopping due to keyboard interrupt")
                break
