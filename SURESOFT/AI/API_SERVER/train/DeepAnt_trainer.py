import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.linalg import norm

from net.DeepAnt_net import DeepAntNet


class DeepAntModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.model = DeepAntNet(n_features)
        self.device = self.get_default_device()

    def get_default_device(self):
        """Pick GPU if available, else CPU"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def to_device(self, data, device):
        """Move tensor(s) to chosen device"""
        if isinstance(data, (list, tuple)):
            return [self.to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)

    def training_step(self, batch):
        x, y = batch
        y_pred = self.model(x)
        loss = self.lossfn(y_pred, y)

        return loss

    def validation_step(self, batch):
        with torch.no_grad():
            x, y = batch
            y_pred = self.model(x)
            loss = self.lossfn(y_pred, y)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        batch_loss = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_loss).mean()

        return {"val_loss": epoch_loss.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}".format(epoch, result["val_loss"]))

    def evaluate(self, val_loader):
        outputs = [
            self.validation_step(self.to_device(batch, self.device))
            for batch in val_loader
        ]
        return self.validation_epoch_end(outputs)

    def fit(
        self, epochs, train_loader, val_loader, lossfn=nn.L1Loss(), lr=1e-5, es_epochs=7
    ):
        self.lossfn = lossfn
        history = []

        path = "./checkpoints/"
        if not os.path.exists(path):
            os.makedirs(path)

        optimizer = Adam(self.model.parameters(), lr=lr)
        scheduler1 = ReduceLROnPlateau(optimizer, "min", patience=3)
        earlyStopping = EarlyStopping(patience=es_epochs, verbose=True)

        for epoch in range(epochs):
            for batch in train_loader:
                batch = self.to_device(batch, self.device)
                loss = self.training_step(batch)
                loss.backward()

                optimizer.step()

            result = self.evaluate(val_loader)
            self.epoch_end(epoch, result)
            history.append(result)
            earlyStopping(
                result["val_loss"],
                {
                    "n_features": self.model.n_feature,
                    "weights": self.model.state_dict(),
                },
                path,
            )
            if earlyStopping.early_stop:
                print("Early stopping")
                break
            scheduler1.step(result["val_loss"])
            print("lr : ", scheduler1.get_last_lr())

        return history

    def testing(self, test_loader):
        results = []
        with torch.no_grad():
            for batch in test_loader:
                batch = self.to_device(batch, self.device)
                x, y = batch
                y_pred = self.model(x)

                results.append(norm(y_pred - y).cpu().detach().item())
        return results


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name="DeepAnT", delta=1e-6):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, model, path):
        score = val_loss

        if self.best_score is None:
            self.best_score = score

            self.save_checkpoint(val_loss, model, path)
        elif score > self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print("Validation loss decreased.  Saving model ...")
        torch.save(model, os.path.join(path, str(self.dataset) + "_checkpoint.pth"))
        self.val_loss_min = val_loss
