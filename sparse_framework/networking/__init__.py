import sys

if sys.version_info >= (3, 8, 10):
    from .tcp_client_latest import TCPClientLatest as TCPClient
    from .tcp_server_latest import TCPServerLatest as TCPServer
else:
    from .tcp_client_legacy import TCPClientLegacy as TCPClient
    from .tcp_server_legacy import TCPServerLegacy as TCPServer

__all__ = ["TCPClient", "TCPServer"]
