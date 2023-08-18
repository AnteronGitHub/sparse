if __name__ == "__main__":
    from . import ModelServer, DiskModelRepository

    ModelServer(model_repository=DiskModelRepository()).start()
