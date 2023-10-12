import asyncio

from torch.autograd import Variable

from sparse_framework import TaskExecutor
from sparse_framework.dl import count_model_parameters, ModelExecutor

class GradientCalculator(TaskExecutor):
    def __init__(self, device, capacity = 0, **args):
        super().__init__(**args)
        self.capacity = capacity
        self.device = device
        self.tasks = set()

    def submit_task(self, input_data, callback):
        executor_task = asyncio.create_task(self.execute_task(input_data, callback))
        self.tasks.add(executor_task)
        executor_task.add_done_callback(self.tasks.discard)

        self.logger.debug(f"Created executor task.")

    async def execute_task(self, input_data: dict, callback) -> dict:
        """Execute a single gradient computation for the offloaded layers."""
        self.logger.debug(f"Executing task.")
        split_layer, labels, model, loss_fn, optimizer = input_data['activation'], \
                                                         input_data['labels'], \
                                                         input_data['model'], \
                                                         input_data['loss_fn'], \
                                                         input_data['optimizer']

        split_layer = Variable(split_layer, requires_grad=True).to(self.device)
        split_layer.retain_grad()

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

        gradient = None if split_layer.grad is None else split_layer.grad.to("cpu").detach()

        callback({ "gradient": gradient, "loss": loss })

