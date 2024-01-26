from torch import nn
from .Encoder import VGGEncoder
from .Decoder import InitDecoder


class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()

        self.encoder_seq = VGGEncoder()
        self.decoder_seq = InitDecoder([512, 512, 256, 128, 64, 1])

        self.out = nn.Conv2d(1, 1, 1, padding = "same")

    def forward(self, x):
        skip_connections = []
        for layer in self.encoder_seq:
            x = layer(x)
            skip_connections.append(x)
            x = nn.MaxPool2d(2)(x)

        for layer, skip in zip(self.decoder_seq, skip_connections[::-1]):
            x = layer(x, skip)

        return self.out(x)