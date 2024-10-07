import uuid

__all__ = ["StreamOperator"]

class StreamOperator:
    """Stream operator processes tuples from an input stream to produce tuples in an output stream.
    """
    def __init__(self, use_batching : bool = True):
        self.id = str(uuid.uuid4())
        self.batch_no = 0
        self.use_batching = use_batching

        self.executor = None

    @property
    def name(self):
        return self.__class__.__name__

    def set_executor(self, executor):
        self.executor = executor

    def buffer_input(self, data_tuple, on_result_received):
        self.executor.buffer_input(self.id, data_tuple, on_result_received, None)

    def call(self, input_tuple):
        pass

