if __name__ == "__main__":
    from . import ModelServer, ModelTrainingRepository

    ModelServer(model_repository=ModelTrainingRepository()).start()
