from graphlib import TopologicalSorter

from .module_repo import ModuleRepository, SparseModule
from .node import SparseSlice
from .protocols import SparseProtocol
from .runtime import SparseRuntime
from .stream_router import StreamRouter

class ClusterOrchestrator(SparseSlice):

    def __init__(self, runtime : SparseRuntime, stream_router : SparseRuntime, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.runtime = runtime
        self.stream_router = stream_router

    def create_deployment(self, source : SparseProtocol, app_dag : dict):
        """Deploys a Sparse application to a cluster.

        The application graph is sorted topologically so that each destination node is deployed before its sources.

        :param app_name: The name of the Sparse application to be deployed.
        :param app_dag: A dictionary representing the Directed Asyclic Graph of application nodes.
        """
        self.logger.debug("Creating deployment for app graph %s", app_dag)

        # Place operators
        # TODO: use explicit operator selectors instead of trying to deploy.
        for stream_selector in TopologicalSorter(app_dag).static_order():
            self.runtime.place_operator(stream_selector)

        # Connect streams
        for stream_selector in TopologicalSorter(app_dag).static_order():
            destinations = app_dag[stream_selector] if stream_selector in app_dag.keys() else set()

            output_stream = self.stream_router.get_stream(stream_alias=stream_selector)
            self.stream_router.connect_to_operators(output_stream, destinations)
