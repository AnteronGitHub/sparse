import os
from dotenv import load_dotenv

class ConfigManager:
    def __init__(self):
        self.upstream_host = None
        self.upstream_port = None
        self.listen_address = None
        self.listen_port = None
        self.model_server_address = None
        self.model_server_port = None

    def load_config(self):
        load_dotenv(dotenv_path=".env")

        self.upstream_host = os.environ.get('MASTER_UPSTREAM_HOST') or '127.0.0.1'
        self.upstream_port = os.environ.get('MASTER_UPSTREAM_PORT') or 50007
        self.listen_address = os.environ.get('WORKER_LISTEN_ADDRESS') or '127.0.0.1'
        self.listen_port = os.environ.get('WORKER_LISTEN_PORT') or 50007
        self.model_server_address = os.environ.get('SPARSE_MODEL_SERVER_ADDRESS') or '127.0.0.1'
        self.model_server_port = os.environ.get('SPARSE_MODEL_SERVER_PORT') or 50006
