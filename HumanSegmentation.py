from Structure.model.Detector import Detector
from torch import nn
import numpy as np
import torch


class BodyDetector(nn.Module):
    def __init__(self, model_path):
        super(BodyDetector, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Detector().to(self.device)
        self.__load_model__(model_path)

    def __load_model__(self, fname):
        self.model.load_state_dict(torch.load(fname))
        self.model.eval()


    def __normalization__(self, x) -> torch.Tensor:
        return ((x - x.mean()) / x.std()).unsqueeze(0)


    def DetectBody(self, x):
        assert len(x.shape) == 3, "Image must have 3 channels (c, h, w)!"
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, device = self.device, dtype = torch.float32)
            x = torch.moveaxis(x, -1, 0)

        x = self.__normalization__(x)
        pred = self.model(x)
        pred = torch.nn.Sigmoid()(pred)

        mask = pred.cpu().detach().numpy()[0]
        mask = np.moveaxis(mask, 0, -1)

        return np.where(mask > (mask.mean() + abs(mask.std() / 2)), 255, 0).astype("uint8")