import models.basic_nn as basic_nn
import models.vgg as vgg


FIRST_SPLIT = {
    "basic": basic_nn.NeuralNetwork_local,
    "vgg": vgg.NeuralNetwork_local,
}

SECOND_SPLIT = {
    "basic": basic_nn.NeuralNetwork_server,
    "vgg": vgg.NeuralNetwork_server,
}
