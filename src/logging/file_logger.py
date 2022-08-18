from datetime import datetime
import os

class FileLogger():
    def __init__(self, data_dir = './data/stats', file_prefix = 'statistics', verbose = False):
        os.makedirs(data_dir, exist_ok=True)
        experiment_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.filepath = os.path.join(data_dir, f'{file_prefix}-{experiment_time}.csv')

        self.verbose = verbose

    def log_row(self, row):
        if self.verbose:
            print(row)
        with open(self.filepath, 'a') as f:
            f.write(row + '\n')

