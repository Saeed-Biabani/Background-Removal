from Structure.model.Detector import Detector
from torchvision.transforms import Compose
from Structure.Losses import DiceLoss
from Structure.Trainer import Trainer
from Structure.DataUtils import *
import torch

root = "path_to_dataset"
model = Detector()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
loss_fn = DiceLoss()
scaler = torch.cuda.amp.GradScaler()
batch_size = 8
num_epoch = 25
transforms = transforms = Compose([
    Resize((224, 224)),
    Rescale(1./255),
    Normalization()
])

trainer = Trainer(
    root,
    model,
    optimizer,
    loss_fn,
    scaler,
    batch_size,
    num_epoch,
    transforms = transforms
)

trainer.train()