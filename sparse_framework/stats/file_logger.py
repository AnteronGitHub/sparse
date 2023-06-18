from datetime import datetime
import os

class FileLogger():
    def __init__(self, benchmark_id, data_dir = '/data/stats', file_prefix = 'statistics'):
        os.makedirs(data_dir, exist_ok=True)
        experiment_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        benchmark_id_formatted = benchmark_id.replace("-", "_")
        self.filepath = os.path.join(data_dir, f'{file_prefix}-{experiment_time}-{benchmark_id_formatted}.csv')

    def log_row(self, stats):
        row = ','.join([str(v) for v in stats])

        with open(self.filepath, 'a') as f:
            f.write(row + '\n')

