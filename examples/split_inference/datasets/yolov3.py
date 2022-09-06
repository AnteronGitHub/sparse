import os
from urllib import request

from utils import get_device, ImageLoading, non_max_suppression, save_detection

samples_url = "https://raw.githubusercontent.com/westerndigitalcorporation/YOLOv3-in-PyTorch/release/data/samples"

class YOLOv3Dataset:
    def __init__(self, data_dir = "./data"):
        self.samples_dir = os.path.join(data_dir, "samples")
        self.sample_files = [
                "dog.jpg",
                "eagle.jpg",
                "giraffe.jpg",
                "herd_of_horses.jpg",
                "img1.jpg",
                "img2.jpg",
                "img3.jpg",
                "img4.jpg",
                "messi.jpg",
                "person.jpg",
        ]
        self.download_files()

    def download_files(self):
        os.makedirs(self.samples_dir, exist_ok=True)
        for sample_file in self.sample_files:
            file_url = os.path.join(samples_url, sample_file)
            target_file = os.path.join(self.samples_dir, sample_file)
            if os.path.exists(target_file):
                print(f"Sample file '{sample_file}' already downloaded.")
            else:
                print(f"Downloading sample file '{sample_file}'.")
                request.urlretrieve(file_url, target_file)

    def get_sample(self, img_size):
        imagePath = os.path.join(self.samples_dir, "dog.jpg")
        return ImageLoading(imagePath, img_size)
