#why logging?
#1. Monitor training process
#2. Debugging Model
#3. Improve Model
#4. Keep records of your experiments so you can come back next time without having to run the same experiment again

#Things you want to capture
#1. Training Loss (going downwards that shows model is learning)
#2. Model Performance training and validation metrics
#3. Hyperparameters

import lightning as L
import torch
from shared_utilities import LightningModel, MNISTDataModule, PyTorchMLP
from watermark import watermark

if __name__ == "__main__":

    print(watermark(packages="torch,lightning", python=True))
    print("Torch CUDA available?", torch.cuda.is_available())

    torch.manual_seed(123)

    # NEW !!!
    dm = MNISTDataModule()

    pytorch_model = PyTorchMLP(num_features=784, num_classes=10)

    lightning_model = LightningModel(model=pytorch_model, learning_rate=0.05)

    trainer = L.Trainer(
        max_epochs=10,
        accelerator="cpu", 
        devices="auto", 
        deterministic=True,
        default_root_dir= "./logger"

    )

    # NEW !!!
    # trainer.fit(model=lightning_model,
    #             train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.fit(model=lightning_model, datamodule=dm)

    train_acc = trainer.validate(dataloaders=dm.train_dataloader())[0]["val_acc"]
    val_acc = trainer.validate(datamodule=dm)[0]["val_acc"]
    test_acc = trainer.test(datamodule=dm)[0]["test_acc"]
    print(
        f"Train Acc {train_acc*100:.2f}%"
        f" | Val Acc {val_acc*100:.2f}%"
        f" | Test Acc {test_acc*100:.2f}%"
    )

#using tensorboard to visualize the logs on CML tensorboard --logdir logger