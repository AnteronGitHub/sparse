import sys

if sys.version_info >= (3, 8, 10):
    from .tcp_client_latest import TCPClientLatest as TCPClient
    from .tcp_server_latest import TCPServerLatest as TCPServer
else:
    from .tcp_client_legacy import TCPClientLegacy as TCPClient
    from .tcp_server_legacy import TCPServerLegacy as TCPServer

from .unix_socket_client import UnixSocketClient
from .unix_socket_server import UnixSocketServer

__all__ = [ "TCPClient",
            "TCPServer",
            "UnixSocketClient",
            "UnixSocketServer" ]
