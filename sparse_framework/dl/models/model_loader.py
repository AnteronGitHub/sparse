import asyncio
import logging
import time

from .model_server import decode_offload_reply, encode_model_request

class ModelLoader():
    def __init__(self,
                 model_server_address : str,
                 model_server_port : int):
        logging.basicConfig(format='[%(asctime)s] %(name)s - %(levelname)s: %(message)s', level=logging.DEBUG)
        self.logger = logging.getLogger("sparse")

        self.model_server_address = model_server_address
        self.model_server_port = model_server_port

    async def stream_model(self, model_name : str, partition, compressionProps, use_compression):
        self.logger.debug(f"Connecting to {self.model_server_address}:{self.model_server_port}...")
        while True:
            try:
                reader, writer = await asyncio.open_connection(self.model_server_address, self.model_server_port)
                break
            except ConnectionRefusedError:
                self.logger.error(f"Unable to connect to upstream host {self.model_server_address}:{self.model_server_port}. Trying again in 5 seconds...")
                time.sleep(5)
            except TimeoutError:
                self.logger.error("Connection to upstream host timed out. Retrying...")

        writer.write(encode_model_request(model_name, partition, compressionProps, use_compression))
        writer.write_eof()
        await writer.drain()

        result_data = await reader.read()
        writer.close()

        return decode_offload_reply(result_data)

    async def deploy_task(self, model_name : str, partition, compressionProps, use_compression):
        task = asyncio.create_task(self.stream_model(model_name, partition, compressionProps, use_compression))
        await task
        return task.result()

    def load_model(self, model_name : str, partition, compressionProps, use_compression):
        return asyncio.run(self.deploy_task(model_name, partition, compressionProps, use_compression))
