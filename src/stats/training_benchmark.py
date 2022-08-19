import argparse

from tqdm import tqdm

from sparse.stats.training_monitor import TrainingMonitor
from sparse.logging.file_logger import FileLogger

class TrainingBenchmark():
    def __init__(self,
                 model_name,
                 batches = 64,
                 batch_size = 64,
                 epochs = 1,
                 log_file_prefix = 'training-benchmark'):
        self.model_name = model_name
        self.batches = batches
        self.batch_size = batch_size
        self.epochs = epochs
        self.log_file_prefix = log_file_prefix

    def start(self):
        self.monitor = TrainingMonitor()
        print(f"Training {self.model_name} model in {self.epochs} epochs with {self.batches*self.batch_size} samples using batch size {self.batch_size}")
        self.logger = FileLogger(file_prefix=f"{self.log_file_prefix}-{self.model_name}-{self.epochs}-{self.batch_size}")
        self.logger.log_row(self.monitor.get_metrics())
        self.logger.log_row(','.join([str(v) for v in self.monitor.read_stats(0)]))
        self.progress_bar = tqdm(total=self.epochs*self.batches*self.batch_size,
                                 unit='samples',
                                 unit_scale=True)

    def add_point(self, processed_samples = 0):
        self.logger.log_row(','.join([str(v) for v in self.monitor.read_stats(processed_samples)]))
        self.progress_bar.update(processed_samples)

    def end(self):
        self.progress_bar.close()
