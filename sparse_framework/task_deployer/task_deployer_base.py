import pickle

class TaskDeployerBase:
    """Class that handles network connections to available worker nodes.
    """

    def __init__(self):
        self.node = None
        self.logger = None

    def set_node(self, node):
        self.node = node
        self.logger = node.logger
        self.logger.info(f"Task deployer using upstream {self.node.config_manager.upstream_host}:{self.node.config_manager.upstream_port}")

    def deploy_task(self, input_data : dict):
        pass

    def encode_request(self, input_data : dict) -> bytes:
        return pickle.dumps(input_data)

    def decode_response(self, result_payload : bytes) -> dict:
        return pickle.loads(result_payload)
