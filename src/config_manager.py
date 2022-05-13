import os

class ConfigManager:
    def load_config(self):
        pass

class MasterConfigManager(ConfigManager):
    def __init__(self):
        self.upstream_host = None
        self.upstream_port = None

    def load_config(self):
        self.upstream_host = os.environ.get('MASTER_UPSTREAM_HOST') or '127.0.0.1'
        self.upstream_port = os.environ.get('MASTER_UPSTREAM_PORT') or 50007

class WorkerConfigManager(ConfigManager):
    def __init__(self):
        self.listen_address = None
        self.listen_port = None

    def load_config(self):
        self.listen_address = os.environ.get('WORKER_LISTEN_ADDRESS') or '127.0.0.1'
        self.listen_port = os.environ.get('WORKER_LISTEN_PORT') or 50007

