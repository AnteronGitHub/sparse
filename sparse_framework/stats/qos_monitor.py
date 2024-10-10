import asyncio

from time import time

from ..runtime import StreamOperator
from ..node import SparseSlice

class OperatorRuntimeStatisticsRecord:
    """Operator runtime statistics record tracks tuple processing latency for a given operator.
    """
    operator_id : str
    source_stream_id : str
    input_buffered_at : float
    result_received_at : float

    def __init__(self, operator_id : str, source_stream_id : str):
        self.operator_id = operator_id
        self.source_stream_id = source_stream_id
        self.input_buffered_at = None
        self.result_received_at = None

    def input_buffered(self):
        self.input_buffered_at = time()

    def result_received(self):
        self.result_received_at = time()

    @property
    def processing_latency(self) -> float:
        """Processing latency for the operator in milliseconds.
        """
        if self.result_received_at is None or self.input_buffered_at is None:
            return None
        return (self.result_received_at - self.input_buffered_at)*1000.0

class OperatorRuntimeStatisticsService:
    def __init__(self):
        self.records = set()

    def get_operator_runtime_statistics_record(self, operator, source):
        """Returns an operator runtime statistics records matching given operator and source stream. If one is not
        already found it will be created.
        """
        for record in self.records:
            if record.operator_id == operator.id and record.source_stream_id == source.stream_id:
                return record

        record = OperatorRuntimeStatisticsRecord(operator.id, source.stream_id)
        self.records.add(record)
        return record

class QoSMonitor(SparseSlice):
    """Quality of Service Monitor Slice maintains a coroutine for monitoring the runtime performance of the node.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.statistics_service = OperatorRuntimeStatisticsService()

    def operator_input_buffered(self, operator : StreamOperator, source):
        record = self.statistics_service.get_operator_runtime_statistics_record(operator, source)
        record.input_buffered()

    def operator_result_received(self, operator : StreamOperator, source):
        record = self.statistics_service.get_operator_runtime_statistics_record(operator, source)
        record.result_received()
        self.logger.info("Operator %s processing latency: %.2f ms", operator, record.processing_latency)
