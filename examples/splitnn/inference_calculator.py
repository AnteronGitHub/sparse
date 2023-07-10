from sparse_framework.dl import ModelExecutor

class InferenceCalculator(ModelExecutor):
    def start(self):
        super().start()
        self.logger.info(f"Inferring with the model.")

    async def execute_task(self, input_data: dict) -> dict:
        """Execute a single forward computation for the offloaded layers."""
        split_layer = input_data['activation']
        pred = self.model(split_layer.to(self.device))

        if self.task_deployer:
            response_data = await self.task_deployer.deploy_task({ 'activation': pred.to("cpu").detach() })
            return response_data
        else:
            self.logger.debug("Not deploying task any further")
            return { 'prediction': pred.to("cpu").detach() }

