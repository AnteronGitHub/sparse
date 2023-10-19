from datetime import datetime
import logging
import os

from .request_statistics_record import RequestStatisticsRecord

class FileLogger():
    def __init__(self, data_dir = '/data/stats'):
        self.logger = logging.getLogger("sparse")

        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.files = {}

    def get_log_file(self, statistics : str):
        log_file_name = f"{statistics}.csv"
        if statistics not in self.files.keys():
            self.files[statistics] = os.path.join(self.data_dir, log_file_name)
            with open(self.files[statistics], 'a') as f:
                f.write(RequestStatisticsRecord.csv_header())
        return self.files[statistics]

    def log_record(self, record):
        file_path = self.get_log_file(type(record).__name__)
        if file_path is not None:
            with open(file_path, 'a') as f:
                f.write(record.to_csv())
