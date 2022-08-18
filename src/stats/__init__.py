from datetime import datetime
import os
import time

class FileLogger():
    def __init__(self, data_dir = './data/stats'):
        os.makedirs(self.data_dir, exist_ok=True)
        experiment_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.filepath = os.path.join(data_dir, f'statistics-{experiment_time}.csv')

    def log_row(self, row):
        print(row)
        with open(self.filepath, 'a') as f:
            f.write(row + '\n')

class Monitor():
    def __init__(self):
        initial_time = None

    def get_metrics(self):
        return 'timestamp'

    def read_stats(self):
        if initial_time is None:
            initial_time = time()

        return time() - initial_time

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

