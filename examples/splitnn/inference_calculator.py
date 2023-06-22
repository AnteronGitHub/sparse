from sparse_framework.dl import ModelExecutor

from serialization import decode_offload_inference_request, \
                          encode_offload_inference_request, \
                          decode_offload_inference_response, \
                          encode_offload_inference_response


class InferenceCalculator(ModelExecutor):
    def start(self):
        super().start()
        self.logger.info(f"Inferring with the model.")

    async def execute_task(self, input_data: bytes) -> bytes:
        """Execute a single forward computation for the offloaded layers."""
        split_layer = decode_offload_inference_request(input_data)
        pred = self.model(split_layer.to(self.device))

        if self.task_deployer:
            offload_input_data = encode_offload_inference_request(pred.to("cpu").detach())
            result_data = await self.task_deployer.deploy_task(offload_input_data)
        else:
            self.logger.debug("Not deploying task any further")
            result_data = encode_offload_inference_response(pred.to("cpu").detach())

        return result_data

