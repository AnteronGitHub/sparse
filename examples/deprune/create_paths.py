from models.yolov3 import Darknet, YOLOv3
import torch
import torch.nn as nn
model = Darknet().to('cpu') #img_size=opt.img_size,
model.load_darknet_weights("models/yolov3/yolov3.weights")
torch.save(model.state_dict(),"models/yolov3/yolov3_all.paths")
