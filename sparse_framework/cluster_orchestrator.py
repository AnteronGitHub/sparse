from .deployment import Deployment
from .node import SparseSlice
from .protocols import SparseProtocol
from .runtime import SparseRuntime
from .stream_api import SparseStream
from .stream_router import StreamRouter

class ClusterOrchestrator(SparseSlice):
    def __init__(self, runtime : SparseRuntime, stream_router : SparseRuntime, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.runtime = runtime
        self.stream_router = stream_router

    def deploy_pipelines(self, streams : set, pipelines : dict, source : SparseStream = None):
        for stream_selector in pipelines.keys():
            if stream_selector in streams:
                output_stream = self.stream_router.get_stream(stream_alias=stream_selector)
            else:
                operator = self.runtime.place_operator(stream_selector)
                if source is None:
                    self.logger.warn("Placed operator '%s' with no input stream", operator)
                else:
                    output_stream = self.stream_router.get_stream()
                    source.connect_to_operator(operator, output_stream)

            destinations = pipelines[stream_selector]
            if type(destinations) == dict:
                self.deploy_pipelines(streams, destinations, output_stream)
            elif type(destinations) == list:
                for selector in destinations:
                    if selector in streams:
                        final_stream = self.stream_router.get_stream(selector)
                        output_stream.connect_to_stream(final_stream)
                    else:
                        self.logger.warn("Leaf operator %s not created", selector)

    def create_deployment(self, deployment : Deployment):
        """Deploys a Sparse pipelines to a cluster.
        """
        self.logger.debug("Creating deployment %s", deployment)

        self.deploy_pipelines(deployment.streams, deployment.pipelines)
