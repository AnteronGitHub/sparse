import uuid

from .io_buffer import SparsePytorchIOBuffer

__all__ = ["StreamOperator"]

class StreamOperator:
    """Stream operator processes tuples from an input stream to produce tuples in an output stream.
    """
    def __init__(self, use_batching : bool = True):
        self.id = str(uuid.uuid4())
        self.batch_no = 0
        self.use_batching = use_batching

        self.runtime = None
        self.memory_buffer = SparsePytorchIOBuffer()

    def __str__(self):
        return self.name

    @property
    def name(self):
        return self.__class__.__name__

    def set_runtime(self, runtime):
        self.runtime = runtime

    def buffer_input(self, input_data, result_callback):
        return self.memory_buffer.buffer_input(input_data, result_callback)

    def dispatch_buffer(self):
        if self.use_batching:
            return self.memory_buffer.dispatch_batch()
        else:
            return self.memory_buffer.pop_input()

    def result_received(self, result, callbacks):
        self.memory_buffer.result_received(result, callbacks, use_batching = self.use_batching)

    def call(self, input_tuple):
        pass

