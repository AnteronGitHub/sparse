import asyncio
import logging
import pickle
import time

class BaseTCPClient():
    def __init__(self, server_address : str, server_port : int):
        logging.basicConfig(format='[%(asctime)s] %(name)s - %(levelname)s: %(message)s', level=logging.DEBUG)
        self.logger = logging.getLogger("sparse")

        self.server_address = server_address
        self.server_port = server_port

    def decode_data(self, payload : bytes) -> dict:
        return pickle.loads(payload)

    def encode_data(self, result_data : dict) -> bytes:
        return pickle.dumps(result_data)

    async def _send_request(self, request_payload : bytes) -> bytes:
        while True:
            try:
                task = asyncio.open_connection(self.server_address, self.server_port)
                reader, writer = await asyncio.wait_for(task, timeout=5)
                break
            except ConnectionRefusedError:
                self.logger.error(f"Unable to connect to TCP server on {self.server_address}:{self.server_port}. Trying again in 5 seconds...")
                await asyncio.sleep(5)
            except TimeoutError:
                self.logger.error("Connection to TCP server on {self.server_address}:{self.server_port} timed out. Retrying...")

        writer.write(request_payload)
        writer.write_eof()
        await writer.drain()

        try:
            result_payload = await reader.read()
            writer.close()

            return result_payload
        except ConnectionResetError:
            return None

    def broken_pipe_error(self):
        self.logger.error(f"Broken pipe error. Re-executing...")

    async def _create_request(self, request_data : dict) -> dict:
        pass

    def process_request(self, request_data : dict) -> dict:
        return asyncio.run(self._create_request(request_data))

