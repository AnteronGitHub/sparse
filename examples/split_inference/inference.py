import torch
from torch.autograd import Variable

from utils import non_max_suppression, save_detection, get_device, ImageLoading


def inference(imagePath, img, model_local, model_server, quantizeDtype = torch.float16, realDtype = torch.float32):

    model_local.eval()
    model_server.eval()
    with torch.no_grad():
        
            X = img.to(device)
            
            # Compute prediction error
            split_vals = model_local(X, local=True)
            detached_split_vals = split_vals.detach()
            quantized_split_vals = detached_split_vals.to(quantizeDtype)
            transfererd_split_vals = quantized_split_vals.detach().to('cuda')
            dequantized_split_vals = transfererd_split_vals.detach().to(realDtype)
            serverInput_split_vals = Variable(dequantized_split_vals, requires_grad=True)    
            pred_temp = model_server(serverInput_split_vals)
            
            pred = pred_temp[0]
            conf_thres = 0.95 #object confidence threshold
            nms_thres = 0.3 #iou thresshold for non-maximum suppression
            detection = non_max_suppression(pred_temp, conf_thres, nms_thres)
            
            img = X
            save_detection(img, imagePath, detection)





compressionProps = {} ### 
compressionProps['feature_compression_factor'] = 4 ### resolution compression factor, compress by how many times
compressionProps['resolution_compression_factor'] = 1 ###layer compression factor, reduce by how many times TBD


img_size = 416
device = get_device()
config_path_local = "config/yolov3_local.cfg"
config_path_server = "config/yolov3_server.cfg"
#config_path_server = "config/yolov3.cfg"
weight_local = "weights/yolov3_local.paths"
weight_server = "weights/yolov3_server.paths"



model1 = NeuralNetwork_local(config_path_local, compressionProps).to(device)
model1.load_state_dict(torch.load(weight_local))
prev_filters = model1.prevfiltersGet()
model2 = NeuralNetwork_server(config_path_server, compressionProps, prev_filters)
print(model2)
model2.load_state_dict(torch.load(weight_server))
model2 = model2.to('cuda')
imagePath = "/data/samples/dog.jpg" 
img = ImageLoading(imagePath,img_size)

inference(imagePath, img, model1, model2)
