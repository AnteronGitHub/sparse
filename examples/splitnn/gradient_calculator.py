from torch.autograd import Variable

from sparse_framework.dl import count_model_parameters, ModelExecutor

class GradientCalculator(ModelExecutor):
    def __init__(self, capacity = 0, **args):
        super().__init__(**args)
        self.capacity = capacity
        self.delayed_save = None

    def start(self):
        super().start()
        self.model.train()
        self.logger.info(f"Training the model.")

    async def execute_task(self, input_data: dict) -> dict:
        """Execute a single gradient computation for the offloaded layers."""
        if self.delayed_save is not None and not self.delayed_save.done():
            self.delayed_save.cancel()

        split_layer, labels, client_capacity = input_data['activation'], input_data['labels'], input_data['capacity']
        split_layer = Variable(split_layer, requires_grad=True).to(self.device)
        split_layer.retain_grad()

        pred = self.model(split_layer)

        if self.task_deployer:
            response_data = await self.task_deployer.deploy_task({ 'activation': pred.to("cpu").detach(),
                                                                   'labels': labels,
                                                                   'capacity' : self.capacity })
            split_grad, loss = response_data['gradient'], response_data['loss']
            split_grad = split_grad.to(self.device)

            self.optimizer.zero_grad()
            pred.backward(split_grad)
            self.optimizer.step()

            if response_data['piggyback_module'] is not None:
                self.model.append(response_data['piggyback_module'])
                self.capacity = 0
                num_parameters = count_model_parameters(self.model)
                self.logger.info(f"Received piggyback module. {num_parameters} local parameters.")
        else:
            loss = self.loss_fn(pred, labels.to(self.device))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss = loss.item()

        if client_capacity > 0:
            piggyback_module = self.model.pop()
            num_parameters = count_model_parameters(self.model)
            self.logger.info(f"Sending piggyback module. {num_parameters} local parameters.")
        else:
            piggyback_module = None

        self.delayed_save = self.node.add_timeout(self.save_model)

        return { "gradient": split_layer.grad.to("cpu").detach(), "loss": loss, "piggyback_module": piggyback_module }

