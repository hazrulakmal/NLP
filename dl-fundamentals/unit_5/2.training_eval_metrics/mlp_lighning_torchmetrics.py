import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from utils import PytorchMLP, get_dataset_loaders, compute_accuracy
from watermark import watermark

# LightningModule that receives the model as input
class LightningModel(pl.LightningModule):
    # Initialize the model and its hyperparameters
    def __init__(self, model, lr=0.01):
        super().__init__()
        self.model = model
        self.lr = lr

        # Computes training accuracy batch by batch conviently (no extra iteration)
        self.train_acc = torchmetrics.Accuracy(task="multiclass",  num_classes=10) 
        self.val_acc = torchmetrics.Accuracy(task="multiclass",  num_classes=10)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)


    # Define forward pass
    def forward(self, x):
        return self.model(x)
    
    def _shared_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        predicted_labels = torch.argmax(logits, dim=1)
        return loss, y, predicted_labels

    # Define training step
    def training_step(self, batch, batch_idx):
        loss, y, predicted_labels = self._shared_step(batch, batch_idx)

        self.log("train_loss", loss)
        # Computes training accuracy batch by batch
        # and logs it after epoch is completed.
        # This is not exactly the same as before since model weights change
        # after each batch; but it's much faster (no extra iteration).
        self.train_acc(predicted_labels, y)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, y, predicted_labels = self._shared_step(batch, batch_idx)
        # Computes validation accuracy and loss batch by batch
        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(predicted_labels, y)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, y, predicted_labels = self._shared_step(batch, batch_idx)
        self.test_acc(predicted_labels, y)
        self.log("test_acc", self.test_acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        return optimizer

if __name__=="__main__":

    print(watermark(packages="torch,lightning", python=True))
    print("Torch CUDA available?", torch.cuda.is_available())

    train_loader, val_loader, test_loader = get_dataset_loaders()

    model = PytorchMLP(num_features=784, num_classes=10, hidden_size=50)
    lightning_model = LightningModel(model, lr=0.05)

    trainer = pl.Trainer(
        max_epochs=10, 
        accelerator="auto",
        devices="auto"
    )

    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # Evaluate model based on test_step
    train_acc = trainer.test(dataloaders=train_loader)[0]["accuracy"]
    val_acc = trainer.test(dataloaders=val_loader)[0]["accuracy"]
    test_acc = trainer.test(dataloaders=test_loader)[0]["accuracy"]

