import asyncio
import logging
import json
import socket

class UnixSocketServer():
    def __init__(self, socket_path : str):
        self.socket_path = socket_path
        logging.basicConfig(format='[%(asctime)s] %(name)s - %(levelname)s: %(message)s', level=logging.INFO)
        self.logger = logging.getLogger("sparse")

    def handle_request(self):
        pass

    async def receive_message(self, reader : asyncio.StreamReader, writer : asyncio.StreamWriter) -> None:
        input_data = await reader.read()
        writer.write("ACK".encode())
        writer.write_eof()
        writer.close()

        request_data = json.loads(input_data.decode())
        self.handle_request(request_data)

    async def run_unix_server(self):
        self.logger.info(f"Starting the monitoring server on '{self.socket_path}'")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server = await asyncio.start_unix_server(self.receive_message, path=self.socket_path)
        await server.serve_forever()

