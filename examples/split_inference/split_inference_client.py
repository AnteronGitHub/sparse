import torch
from torch import nn

from sparse.roles.master import Master

from models import NeuralNetwork_local
from serialization import encode_offload_request, decode_offload_response
from utils import get_device, ImageLoading, non_max_suppression, save_detection


class SplitTrainingClient(Master, imagePath, img_size, config_path_local, weight_local):
    def __init__(self, model_kind: str = "basic"):
        super().__init__()
        self.device = get_device()
        self.model = NeuralNetwork_local(config_path_local)
        self.model.load_state_dict(torch.load(weight_local))

        self.imagePath = imagePath
        self.img_size = img_size
        self.img = ImageLoading(self.imagePath, self.img_size)



    def infer(self, epochs: int = 5):
        self.logger.info(
            f"Starting training using {self.device} for local computations"
        )

        # Transfer model to device
        model = self.model.to(self.device)

        with torch.no_grad():
        
            self.logger.info(f"--------- inferring ----------")

            # Transfer training data to device memory
            X = X.to(self.device)

            # Local forward propagation
            split_vals = model(X)

            # Offloaded layers
            input_data = encode_offload_request(split_vals.to("cpu").detach(), y)
            result_data = self.task_deployer.deploy_task(input_data)

            #post process layers
            pred = decode_offload_response(result_data)
            conf_thres = 0.95 #object confidence threshold
            nms_thres = 0.3 #iou thresshold for non-maximum suppression
            detection = non_max_suppression(pred, conf_thres, nms_thres)
            
            img = X
            save_detection(img, imagePath, detection)

        self.logger.info("Done!")


if __name__ == "__main__":
    config_path_local = "config/yolov3_local.cfg"
    weight_local = "weights/yolov3_local.paths"
    img_size = 416
    imagePath = "samples/dog.jpg" 
    SplitTrainingClient(imagePath, img_size, config_path_local, weight_local).infer()