from .rx_pipe_base import RXPipeBase

# Select task deployer implementation based on the Python compiler version
import sys

if sys.version_info >= (3, 8, 10):
    from .rx_pipe_latest import RXPipeLatest as RXPipe
else:
    from .rx_pipe_legacy import RXPipeLegacy as RXPipe

__all__ = ["RXPipeBase", "RXPipe"]
