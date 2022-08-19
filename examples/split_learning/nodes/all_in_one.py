import torch
from torch.utils.data import DataLoader

from sparse.stats.training_benchmark import TrainingBenchmark

from models.vgg import VGG_unsplit

class AllInOne():
    def __init__(self, dataset, classes, model, loss_fn, optimizer):
        self.dataset = dataset
        self.classes = classes
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def train(self, batches, batch_size, epochs, log_file_prefix):
        benchmark = TrainingBenchmark(model_name=type(self.model).__name__,
                                      batches=batches,
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      log_file_prefix=log_file_prefix)

        print(f"Using {self.device} for processing")

        self.model.to(self.device)
        benchmark.start()
        for t in range(epochs):
            for batch, (X, y) in enumerate(DataLoader(self.dataset, batch_size)):
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)

                loss = self.loss_fn(pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                benchmark.add_point(len(X))
                if batch + 1 >= batches:
                    break

        benchmark.end()
