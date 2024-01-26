from torch.utils.data import DataLoader
from Structure.DataUtils import *
import pathlib
import torch
import tqdm
import os

class Trainer:
    def __init__(
        self,
        root : str,
        model : torch.nn.Module,
        optimizer : torch.nn,
        loss_fn : torch.nn,
        scaler : torch.nn,
        batch_size : int,
        num_epoch : int,
        save_path = pathlib.Path(os.path.join("model_weights", "bgrm.pth")),
        transforms = None,
    ):
        """
        Objective:
            - training process handleing.\n
        Args:
            - `root` path to dataset.
            - `model` torch model.
            - `optimizer` optimizer for model optimization while training.
            - `loss_fn` loss function for calculate loss while training.
            - `sacle` torch scale.
            - `batch_size` number of batches.
            - `num_epoch` number of epochs.
            - `transforms` train data transformation function.
        """
        assert torch.cuda.is_available(), "cuda is not available!"
        self.device = torch.device("cuda")
        self.model = model.to(self.device)

        print(f"Start training on {self.device} <{torch.cuda.get_device_name(self.device)}>")

        self.root = root
        self.transforms = transforms
        self.batch_size = batch_size
        self.trainDataLoader = self.__load_dataset__()
        
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scaler = scaler
        
        self.num_epoch = num_epoch
        
        self.save_path = save_path
        self.save_path.parent.mkdir(exist_ok = True)


    def __load_dataset__(self):
        ds = DataGenerator(
            self.root,
            transforms = self.transforms
        )
        return DataLoader(ds, self.batch_size, True)


    def __train_one_epoch__(self, epoch):
        self.model.train()
        loop = tqdm.tqdm(self.trainDataLoader, colour = "green")
        for X, Y in loop:
            X = X.float().to(self.device)
            Y = Y.float().to(self.device)

            with torch.cuda.amp.autocast():
                pred = self.model(X)
                loss = self.__calculate_loss__(pred, Y)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loop.set_postfix(loss = loss.item(), epoch = epoch)


    def __calculate_loss__(self, y_true, y_pred):
        return self.loss_fn(y_true, y_pred)

    
    def __save_model__(self):
        torch.save(self.model.state_dict(), self.save_path)

    
    def train(self):
        for epoch in range(1, self.num_epoch+1):
            self.__train_one_epoch__(epoch)
            self.__save_model__()