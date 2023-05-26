if __name__ == '__main__':
    from sparse_framework.dl import ModelServer
    from models import ModelTrainingRepository

    ModelServer(model_repository=ModelTrainingRepository()).start()
