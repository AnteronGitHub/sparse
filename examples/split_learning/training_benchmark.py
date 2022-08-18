import argparse

from tqdm import tqdm

from sparse.stats.training_monitor import TrainingMonitor
from sparse.logging.file_logger import FileLogger

class TrainingBenchmark():
    def __init__(self, model_name):
        self._parse_arguments()
        self.model_name = model_name

    def _parse_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--batches', default=64, type=int)
        parser.add_argument('--batch-size', default=64, type=int)
        parser.add_argument('--epochs', default=1, type=int)
        parser.add_argument('--log-file-prefix', default='training-benchmark', type=str)
        self.arguments = parser.parse_args()

    def start(self):
        self.monitor = TrainingMonitor()
        print(f"Training {self.model_name} model in {self.arguments.epochs} epochs with {self.arguments.batches*self.arguments.batch_size} samples using batch size {self.arguments.batch_size}")
        self.logger = FileLogger(file_prefix=f"{self.arguments.log_file_prefix}-{self.model_name}-{self.arguments.epochs}-{self.arguments.batch_size}")
        self.logger.log_row(self.monitor.get_metrics())
        self.logger.log_row(','.join([str(v) for v in self.monitor.read_stats(0)]))
        self.progress_bar = tqdm(total=self.arguments.epochs*self.arguments.batches*self.arguments.batch_size,
                                 unit='samples',
                                 unit_scale=True)

    def add_point(self, processed_samples = 0):
        self.logger.log_row(','.join([str(v) for v in self.monitor.read_stats(processed_samples)]))
        self.progress_bar.update(processed_samples)

    def end(self):
        self.progress_bar.close()
