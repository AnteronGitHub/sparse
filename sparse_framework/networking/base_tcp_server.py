import asyncio
import logging
import pickle
import time

class BaseTCPServer():
    """asyncio TCP server implementation that can be extended for different applications. Includes generic
    request/response serialization and encoding.
    """

    def __init__(self, listen_address : str = '0.0.0.0', listen_port : int = 50006):
        logging.basicConfig(format='[%(asctime)s] %(name)s - %(levelname)s: %(message)s', level=logging.INFO)
        self.logger = logging.getLogger("sparse")

        self.listen_address = listen_address
        self.listen_port = listen_port

    async def handle_request(self, input_data : dict, request_context : dict) -> dict:
        return input_data, request_context

    def decode_request(self, payload : bytes) -> dict:
        return pickle.loads(payload)

    def encode_response(self, result_data : dict) -> bytes:
        return pickle.dumps(result_data)

    def request_processed(self, request_context : dict, processing_time : float):
        self.logger.info(f"Processed Request in {processing_time} seconds.")

    async def _connection_callback(self, reader : asyncio.StreamReader, writer : asyncio.StreamWriter) -> None:
        message_received_at = time.time()
        request_context = {}

        input_payload = await reader.read()

        input_data = self.decode_request(input_payload)

        output_data, request_context = await self.handle_request(input_data, request_context)

        output_payload = self.encode_response(output_data)

        writer.write(output_payload)

        # TODO: Come up with a better workaround for IO errors. Ignore for now...
        try:
            await writer.drain()
        except BrokenPipeError:
            self.logger.info("Broken pipe during response stream. Ignoring...")
            pass

        writer.close()

        self.request_processed(request_context, processing_time=time.time() - message_received_at)

    def start(self):
        pass

