from torch import nn
from torchvision.models import vgg16, VGG16_Weights


def VGGEncoder():
    weights = VGG16_Weights.DEFAULT
    base_model = vgg16(weights=weights)
    base_model.training = False
        
    encoder_seq =  nn.ModuleList()
    moduls = nn.Sequential()
    for layer in list(base_model.features.children()):
        if isinstance(layer, nn.modules.pooling.MaxPool2d):
            encoder_seq.append(moduls)
            moduls = nn.Sequential()
        else:
            moduls.append(layer)
    return encoder_seq