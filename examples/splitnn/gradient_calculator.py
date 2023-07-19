from torch.autograd import Variable

from sparse_framework.dl import ModelExecutor

class GradientCalculator(ModelExecutor):
    def start(self):
        super().start()
        self.model.train()
        self.logger.info(f"Training the model.")
        self.delayed_save = None

    async def execute_task(self, input_data: dict) -> dict:
        """Execute a single gradient computation for the offloaded layers."""
        if self.delayed_save is not None and not self.delayed_save.done():
            self.delayed_save.cancel()

        split_layer, labels = input_data['activation'], input_data['labels']
        split_layer = Variable(split_layer, requires_grad=True).to(self.device)
        split_layer.retain_grad()

        pred = self.model(split_layer)

        if self.task_deployer:
            response_data = await self.task_deployer.deploy_task({ 'activation': pred.to("cpu").detach(), 'labels': labels })
            split_grad, loss = response_data['gradient'], response_data['loss']
            split_grad = split_grad.to(self.device)

            self.optimizer.zero_grad()
            pred.backward(split_grad)
            self.optimizer.step()
        else:
            loss = self.loss_fn(pred, labels.to(self.device))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss = loss.item()

        self.delayed_save = self.node.add_timeout(self.save_model)

        return { "gradient": split_layer.grad.to("cpu").detach(), "loss": loss }

