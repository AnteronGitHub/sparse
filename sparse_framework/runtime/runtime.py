import asyncio

from ..module_repo import ModuleRepository, OperatorNotFoundError
from ..node import SparseSlice
from ..stats import QoSMonitor

from .operator import StreamOperator
from .task_executor import SparseTaskExecutor

class SparseRuntime(SparseSlice):
    """Sparse Runtime maintains task executor, and the associated memory manager, for executing stream
    application operations locally.
    """
    def __init__(self, module_repo : ModuleRepository, qos_monitor : QoSMonitor, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.module_repo = module_repo
        self.qos_monitor = qos_monitor

        self.executor = None
        self.operators = set()

    def get_futures(self, futures):
        self.executor = SparseTaskExecutor()

        futures.append(self.executor.start())

        return futures

    def place_operator(self, operator_name : str):
        """Places a stream operator to the local runtime.
        """
        for operator in self.operators:
            if operator.name == operator_name:
                return operator

        try:

            operator_factory = self.module_repo.get_operator_factory(operator_name)

            o = operator_factory()
            o.set_runtime(self)
            self.executor.add_operator(o)
            self.operators.add(o)

            self.logger.info("Placed operator %s", o)

            return o
        except OperatorNotFoundError as e:
            self.logger.warn(e)

    def call_operator(self, operator : StreamOperator, source, input_tuple, output):
        self.executor.buffer_input(operator.id,
                                   input_tuple,
                                   lambda output_tuple: self.result_received(operator,
                                                                             source,
                                                                             output_tuple,
                                                                             output))
        self.qos_monitor.operator_input_buffered(operator, source)

    def result_received(self, operator : StreamOperator, source, output_tuple, output):
        self.qos_monitor.operator_result_received(operator, source)
        output.emit(output_tuple)

    def find_operator(self, operator_name : str):
        for operator in self.operators:
            if operator.name == operator_name:
                return operator

        return None

    def sync_received(self, protocol, stream_id, sync):
        self.logger.debug(f"Received {sync} s sync")
        if self.source is not None:
            if (self.source.no_samples > 0):
                offload_latency = protocol.request_statistics.get_offload_latency(protocol.current_record)

                if not self.source.use_scheduling:
                    sync = 0.0

                target_latency = self.source.target_latency

                loop = asyncio.get_running_loop()
                loop.call_later(target_latency-offload_latency + sync if target_latency > offload_latency else 0, self.source.emit)
            else:
                protocol.transport.close()
