import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from net.USAD_net import Encoder, Decoder


class UsadModel(nn.Module):
    def __init__(self, w_size, z_size):
        super().__init__()
        self.encoder = Encoder(w_size, z_size)
        self.decoder1 = Decoder(z_size, w_size)
        self.decoder2 = Decoder(z_size, w_size)
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

    def training_step(self, batch, n):
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        loss1 = 1 / n * torch.mean((batch - w1) ** 2) + (1 - 1 / n) * torch.mean(
            (batch - w3) ** 2
        )
        loss2 = 1 / n * torch.mean((batch - w2) ** 2) - (1 - 1 / n) * torch.mean(
            (batch - w3) ** 2
        )
        return loss1, loss2

    def validation_step(self, batch, n):
        with torch.no_grad():
            z = self.encoder(batch)
            w1 = self.decoder1(z)
            w2 = self.decoder2(z)
            w3 = self.decoder2(self.encoder(w1))
            loss1 = 1 / n * torch.mean((batch - w1) ** 2) + (1 - 1 / n) * torch.mean(
                (batch - w3) ** 2
            )
            loss2 = 1 / n * torch.mean((batch - w2) ** 2) - (1 - 1 / n) * torch.mean(
                (batch - w3) ** 2
            )
        return {"val_loss1": loss1, "val_loss2": loss2}

    def validation_epoch_end(self, outputs):
        batch_losses1 = [x["val_loss1"] for x in outputs]
        epoch_loss1 = torch.stack(batch_losses1).mean()
        batch_losses2 = [x["val_loss2"] for x in outputs]
        epoch_loss2 = torch.stack(batch_losses2).mean()
        return {"val_loss1": epoch_loss1.item(), "val_loss2": epoch_loss2.item()}

    def epoch_end(self, epoch, result):
        print(
            "Epoch [{}], val_loss1: {:.4f}, val_loss2: {:.4f}".format(
                epoch, result["val_loss1"], result["val_loss2"]
            )
        )

    def evaluate(self, val_loader, n):
        outputs = [
            self.validation_step(self.to_device(batch, self.device), n)
            for [batch] in val_loader
        ]
        return self.validation_epoch_end(outputs)

    def fit(self, epochs, train_loader, val_loader, lr: float = 1e-4, es_epochs=7):
        history = []
        path = "./checkpoints/"
        if not os.path.exists(path):
            os.makedirs(path)

        optimizer1 = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder1.parameters()), lr=lr
        )
        optimizer2 = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder2.parameters()), lr=lr
        )
        scheduler1 = MultiStepLR(
            optimizer1,
            milestones=[(epochs) // 5, (epochs) // 3, (epochs) // 2],
            gamma=0.5,
        )
        scheduler2 = MultiStepLR(
            optimizer2,
            milestones=[(epochs) // 5, (epochs) // 3, (epochs) // 2],
            gamma=0.5,
        )
        earlyStopping = EarlyStopping(patience=es_epochs, verbose=True)

        for epoch in range(epochs):
            for [batch] in train_loader:
                batch = self.to_device(batch, self.device)

                # Train AE1
                loss1, loss2 = self.training_step(batch, epoch + 1)
                loss1.backward()
                optimizer1.step()
                optimizer1.zero_grad()

                # Train AE2
                loss1, loss2 = self.training_step(batch, epoch + 1)
                loss2.backward()
                optimizer2.step()
                optimizer2.zero_grad()

            result = self.evaluate(val_loader, epoch + 1)

            self.epoch_end(epoch, result)

            history.append(result)

            earlyStopping(
                result["val_loss1"],
                result["val_loss2"],
                {
                    "encoder": self.encoder.state_dict(),
                    "decoder1": self.decoder1.state_dict(),
                    "decoder2": self.decoder2.state_dict(),
                },
                path,
            )
            if earlyStopping.early_stop:
                print("Early stopping")
                break
            scheduler1.step()
            scheduler2.step()
            print("lr : ", scheduler1.get_last_lr(), scheduler2.get_last_lr())

        return history

    def testing(self, test_loader, alpha=0.5, beta=0.5):
        results = []
        with torch.no_grad():
            for [batch] in test_loader:
                batch = self.to_device(batch, self.device)
                w1 = self.decoder1(self.encoder(batch))
                w2 = self.decoder2(self.encoder(w1))
                anomaly_score = alpha * torch.mean(
                    (batch - w1) ** 2, axis=1
                ) + beta * torch.mean((batch - w2) ** 2, axis=1)
                results.append(anomaly_score.cpu().detach().item())
        return results


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name="USAD", delta=1e-6):
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

    def __call__(self, val_loss, val_loss2, model, path):
        score = val_loss
        score2 = val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif (
            score < self.best_score + self.delta
            and score2 > self.best_score2 + self.delta
        ):
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f"Validation loss decreased.  Saving model ...")
        torch.save(model, os.path.join(path, str(self.dataset) + "_checkpoint.pth"))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2
