import torch
from tqdm import tqdm

from models.yolov3 import YOLOv3
from utils import get_device, non_max_suppression, save_detection
from datasets.yolov3 import YOLOv3Dataset

from benchmark import parse_arguments

class AllInOne():
    def __init__(self):
        compressionProps = {}
        compressionProps['feature_compression_factor'] = 4
        compressionProps['resolution_compression_factor'] = 1

        self.model = YOLOv3(compressionProps = compressionProps)
        self.dataset = YOLOv3Dataset()
        self.device = get_device()

    def benchmark(self, inferences_to_be_run = 100, save_results = False):
        print(f"Running {inferences_to_be_run} inference tasks")

        self.model.to(self.device)
        progress_bar = tqdm(total=inferences_to_be_run,
                            unit='inferences',
                            unit_scale=True)
        with torch.no_grad():
            inferences_ran = 0
            while inferences_ran < inferences_to_be_run:
                # Load image to processor memory
                X = self.dataset.get_sample(self.model.img_size).to(self.device)

                # Local forward propagation
                pred = self.model(X)

                #post process layers
                conf_thres = 0.95 #object confidence threshold
                nms_thres = 0.3 #iou thresshold for non-maximum suppression
                detection = non_max_suppression(pred, conf_thres, nms_thres)

                if save_results:
                    save_detection(X, imagePath, detection)
                    print("Saved detection result")

                inferences_ran += 1
                progress_bar.update(1)

        progress_bar.close()

if __name__ == "__main__":
    args = parse_arguments()
    AllInOne().benchmark(inferences_to_be_run = args.inferences_to_be_run)
