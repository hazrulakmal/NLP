import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import PytorchMLP, get_dataset_loaders, compute_accuracy
from watermark import watermark

# LightningModule that receives the model as input
class LightningModel(pl.LightningModule):
    def __init__(self, model, lr=0.01):
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("val_loss", loss)
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

    traiin_acc = compute_accuracy(model, train_loader)
    val_acc = compute_accuracy(model, val_loader)
    test_acc = compute_accuracy(model, test_loader)
    print(
        f"Train Acc {train_acc*100:.2f}%"
        f" | Val Acc {val_acc*100:.2f}%"
        f" | Test Acc {test_acc*100:.2f}%"
    )

