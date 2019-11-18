import torch
from dataset import UrbanSound8KDataset

train_loader = torch.utils.data.DataLoader( 
    UrbanSound8KDataset(‘UrbanSound8K_train.pkl’, mode), 
    batch_size=32,
    shuffle=True, 
    num_workers=8,
    pin_memory=True) 

 val_loader = torch.utils.data.DataLoader( 
    UrbanSound8KDataset(‘UrbanSound8K_test.pkl’, mode), 
    batch_size=32,
    shuffle=False, 
    num_workers=8,
    pin_memory=True)

for i, (input, target, filename) in enumerate(train_loader):
#           training code

for i, (input, target, filename) in enumerate(val_loader):
#           validation code
