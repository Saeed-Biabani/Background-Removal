from torchvision.transforms import functional as F
from torchvision.transforms import Grayscale
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from os import path
import torch


class DataGenerator(Dataset):
    def __init__(self, root, transforms = None):
        super(DataGenerator, self).__init__()
        self.root = root
        
        self.MaskDir = Path(path.join(self.root, "Masks"))
        self.ImageDir = Path(path.join(self.root, "Images"))

        assert self.MaskDir.exists() and self.ImageDir.exists()

        self.Images = list(self.ImageDir.glob("*.png"))
        assert len(self.Images) > 0

        self.transforms = transforms


    def __len__(self):
        return len(self.Images)


    def __loadimage__(self, fname):
        img = Grayscale()(io.read_image(str(fname)))
        return img.repeat(3, 1, 1)


    def __loadmask__(self, fname):
        mask = Grayscale()(io.read_image(str(fname)))
        return torch.where(mask != 0, 255, 0)


    def __getitem__(self, indx):
        if torch.is_tensor(indx):
            idx = idx.tolist()
            
        ImageName = self.Images[indx]        
        image = self.__loadimage__(ImageName)
        
        MaskName = ImageName.name.replace(".png", "_mask.png")
        mask = self.__loadmask__(path.join(self.MaskDir, MaskName))
        
        if self.transforms:
            sample = {"image" : image, "mask" : mask}
            augmentation = self.transforms(sample)
            image = augmentation["image"]
            mask = augmentation["mask"]
        
        return image, mask



class Normalization(object):
    def __call__(self, sample : dict) -> dict:
        image = sample["image"]
        image = (image.to(torch.float32) - torch.mean(image.to(torch.float32))) / torch.std(image.to(torch.float32))

        return {"image" : image, "mask" : sample["mask"]}

class Resize(object):
    def __init__(self, to_size):
        self.to_size = to_size

    def __resizeimage__(self, image : torch.Tensor):
        return F.resize(image, self.to_size, interpolation = F.InterpolationMode.NEAREST, antialias = True)

    def __resizemask__(self, mask : torch.Tensor):
        return F.resize(mask, self.to_size, interpolation = F.InterpolationMode.NEAREST, antialias = True)

    def __call__(self, sample) -> dict:
        mask = self.__resizemask__(sample["mask"])
        img = self.__resizeimage__(sample["image"])

        return {"image" : img, "mask" : mask}


class Rescale(object):
    def __init__(self, img_scale):
        self.img_scale = img_scale

    def __call__(self, sample : dict) -> dict:
        image = sample["image"].float() * self.img_scale
        mask = sample["mask"].float() * self.img_scale

        return {"image" : image, "mask" : mask}