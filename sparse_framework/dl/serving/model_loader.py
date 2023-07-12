from sparse_framework.networking import TCPClient

class ModelLoader(TCPClient):
    def load_model(self, model_name : str, partition : str):
        response_data = self.process_request({ 'model_name': model_name, 'partition': partition })
        return response_data['model'], response_data['loss_fn'], response_data['optimizer']
