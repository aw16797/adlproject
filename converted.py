import torch
import time
import argparse
from pathlib import Path
from multiprocessing import cpu_count

import torch
import torchvision
import torch.backends.cudnn
import numpy as np
import torchvision.transforms as trfm
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import UrbanSound8KDataset

torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def main(args):
    tensor = trfm.ToTensor()
    transforms = [tensor]

    args.dataset_root.mkdir(parents=True, exist_ok=True)

    composed = Compose(transforms)

    train_loader = torch.utils.data.DataLoader( 
        UrbanSound8KDataset(‘UrbanSound8K_train.pkl’, mode), 
        batch_size=32,
        shuffle=True, 
        num_workers=8,
        pin_memory=True,
    ) 

     val_loader = torch.utils.data.DataLoader( 
        UrbanSound8KDataset(‘UrbanSound8K_test.pkl’, mode), 
        batch_size=32,
        shuffle=False, 
        num_workers=8,
        pin_memory=True,
    )

    for i, (input, target, filename) in enumerate(train_loader):
    #           training code


    for i, (input, target, filename) in enumerate(val_loader):
    #           validation code


    model = CNN(height=32, width=32, channels=3, class_count=10, dropout=0.9)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), args.learning_rate, momentum=0.9)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )

    trainer = Trainer(
        model, train_loader,  val_loader, criterion, optimizer, summary_writer, DEVICE
    )

    trainer.train(epochs=1, 2, 10, 10)

    summary_writer.close()
}

class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count: int, dropout: float):
        super().__init__()
        self.input_shape = ImageShape(height=height, width=width, channels=channels)
        self.class_count = class_count

        self.dropout = nn.Dropout(dropout)

        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=32,
            kernel_size=(5, 5),
            padding=(2, 2),
        )
        self.initialise_layer(self.conv1)
        self.bn32 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        ## TASK 2-1: Define the second convolutional layer and initialise its parameters
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(5, 5),
            padding=(2, 2),
        )
        self.initialise_layer(self.conv2)
        self.bn64 = nn.BatchNorm2d(64)

        ## TASK 3-1: Define the second pooling layer
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        ## TASK 5-1: Define the first FC layer and initialise its parameters
        self.fc1 = nn.Linear(4096, 1024)
        self.initialise_layer(self.fc1)
        self.bnFC = nn.BatchNorm1d(1024)

        ## TASK 6-1: Define the last FC layer and initialise its parameters
        self.fc2 = nn.Linear(1024, 10)
        self.initialise_layer(self.fc1)


    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn32(self.conv1(images)))
        x = self.pool1(x)

        ## TASK 2-2: Pass x through the second convolutional layer
        x = F.relu(self.bn64(self.conv2(x)))

        ## TASK 3-2: Pass x through the second pooling layer
        x = self.pool2(x)

        ## TASK 4: Flatten the output of the pooling layer so it is of shape
        ##         (batch_size, 4096)
        x = torch.flatten(x,1)

        ## TASK 5-2: Pass x through the first fully connected layer
        x = F.relu(self.bnFC(self.fc1(self.dropout(x))))

        ## TASK 6-2: Pass x through the last fully connected layer
        x = self.fc2(self.dropout(x))

        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)

def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)

def compute_pca(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
):
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)

    # stores total number of examples for each class
    class_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

    # stores total number of correct predictions for each class
    correct_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

    # stores accuracy for each class
    pca_dict = {0:0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0, 6:0.0, 7:0.0, 8:0.0, 9:0.0}

    for i in range(0,len(labels)-1):
        class_dict[labels[i]] += 1
        if labels[i] == preds[i]:
            correct_dict[labels[i]] += 1

    for key, val in pca_dict.items():
        pca_dict[key] = (correct_dict[key]/class_dict[key])

    return pca_dict

def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = (
      f"CNN_bn_"
      f"dropout={args.dropout}_"
      f"bs={args.batch_size}_"
      f"lr={args.learning_rate}_"
      f"momentum=0.9_"
      f"perspective={args.data_aug_perspective}_" +
      f"brightness={args.data_aug_brightness}_" +
      ("hflip_" if args.data_aug_hflip else "") +
      f"run_"
    )
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


if __name__ == "__main__":
    main(parser.parse_args())
