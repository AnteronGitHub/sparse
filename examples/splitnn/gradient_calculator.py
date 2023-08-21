from torch.autograd import Variable

from sparse_framework.dl import count_model_parameters, ModelExecutor

class GradientCalculator(ModelExecutor):
    def __init__(self, capacity = 0, **args):
        super().__init__(**args)
        self.capacity = capacity
        self.delayed_save = None

    def start(self):
        super().start()
        self.logger.info(f"Training the model.")

    async def execute_task(self, input_data: dict) -> dict:
        """Execute a single gradient computation for the offloaded layers."""
        if self.delayed_save is not None and not self.delayed_save.done():
            self.delayed_save.cancel()

        split_layer, labels, model_meta_data, client_capacity = input_data['activation'], \
                                                                input_data['labels'], \
                                                                input_data['model_meta_data'], \
                                                                input_data['capacity']
        split_layer = Variable(split_layer, requires_grad=True).to(self.device)
        split_layer.retain_grad()

        model, loss_fn, optimizer = await self.model_repository.get_model(model_meta_data)
        model.train()
        pred = model(split_layer)

        if self.task_deployer:
            response_data = await self.task_deployer.deploy_task({ 'activation': pred.to("cpu").detach(),
                                                                   'labels': labels,
                                                                   'model_meta_data': model_meta_data,
                                                                   'capacity' : self.capacity })
            split_grad, loss = response_data['gradient'], response_data['loss']

            if optimizer is not None:
                split_grad = split_grad.to(self.device)
                optimizer.zero_grad()
                pred.backward(split_grad)
                optimizer.step()

            if response_data['piggyback_module'] is not None:
                model.append(response_data['piggyback_module'])
                self.capacity = 0
                num_parameters = count_model_parameters(model)
                self.logger.info(f"Received piggyback module. {num_parameters} local parameters.")
        else:
            loss = loss_fn(pred, labels.to(self.device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.item()

        if client_capacity > 0:
            piggyback_module = model.pop()
            num_parameters = count_model_parameters(model)
            self.logger.info(f"Sending piggyback module. {num_parameters} local parameters.")
        else:
            piggyback_module = None

        self.delayed_save = self.node.add_timeout(self.save_model, model_meta_data)

        gradient = None if split_layer.grad is None else split_layer.grad.to("cpu").detach()

        return { "gradient": gradient, "loss": loss, "piggyback_module": piggyback_module }

