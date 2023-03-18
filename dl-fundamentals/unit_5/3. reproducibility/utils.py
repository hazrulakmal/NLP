import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision.datasets import MNIST
from torchvision import transforms

class PytorchMLP(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_size=50):
        super().__init__()

        self.all_layers = torch.nn.Sequential(
            torch.nn.Linear(num_features, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        logits = self.all_layers(x)
        return logits
    
def get_dataset_loaders(download=True):
    train_dataset = MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=download) #60k
    test_dataset = MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=download) #10k

    train_dataset, val_dataset = random_split(train_dataset, [55000, 5000])

    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=64, 
                              shuffle=True,
                              num_workers=0,
                              )
    
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=64,
                            shuffle=False,
                            num_workers=0,
                            )
    
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=64,
                            shuffle=False,
                            num_workers=0,
                            )
    
    return train_loader, val_loader, test_loader

def compute_accuracy(model, data_loader, device=None):
    # why do we compute accuracy batch by batch? - memory limitation reasons 
    # (unable to load the whole big dataset into memory)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.eval()
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
        
        features = features.to(device)
        targets = targets.to(device)
        
        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)

        compare = predictions == targets
        correct_pred += torch.sum(compare)
        num_examples += targets.size(0)

    return correct_pred.float()/num_examples 



    