from graphlib import TopologicalSorter

from .module_repo import ModuleRepository, SparseModule
from .node import SparseSlice
from .protocols import SparseProtocol
from .runtime import SparseRuntime
from .stream_router import StreamRouter

class ClusterOrchestrator(SparseSlice):

    def __init__(self, runtime : SparseRuntime, module_repo : ModuleRepository, stream_router : SparseRuntime, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.runtime = runtime
        self.module_repo = module_repo
        self.stream_router = stream_router

    def deploy_operator(self, operator_name : str):
        """Deploys a Sparse operator to a cluster node from a local module.
        """
        self.logger.debug("Deploying operator '%s'", operator_name)
        operator_factory = self.module_repo.get_operator_factory(operator_name)

        if operator_factory is None:
            return None

        operator = self.runtime.place_operator(operator_factory)

        return operator

    def create_deployment(self, source : SparseProtocol, app_dag : dict):
        """Deploys a Sparse application to a cluster.

        The application graph is sorted topologically so that each destination node is deployed before its sources.

        :param app_name: The name of the Sparse application to be deployed.
        :param app_dag: A dictionary representing the Directed Asyclic Graph of application nodes.
        """
        self.logger.debug("Creating deployment for app graph %s", app_dag)

        # Place operators
        for stream_selector in TopologicalSorter(app_dag).static_order():
            self.deploy_operator(stream_selector)

        # Connect streams
        for stream_selector in TopologicalSorter(app_dag).static_order():
            destinations = app_dag[stream_selector] if stream_selector in app_dag.keys() else set()

            output_stream = self.stream_router.get_stream(stream_alias=stream_selector)
            self.stream_router.connect_to_operators(output_stream, destinations)
