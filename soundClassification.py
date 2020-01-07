import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import torchvision.datasets
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as trfm

import argparse
from pathlib import Path
from dataset import UrbanSound8KDataset
import pickle

testdata = pickle.load(open(UrbanSound8KDataset(‘UrbanSound8K_train.pkl’, mode), 'rb'))
mfcc = testdata[0]['features']['mfcc']
print('MFCC')
print(mfcc.shape)

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Train an Environment Sound Classification",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
default_dataset_dir = Path.home() / ".cache" / "torch" / "datasets"
parser.add_argument("--dataset-root", default=default_dataset_dir)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=0.001, type=float, help="Learning rate")
parser.add_argument("--batch-size", default=32, type=int, help="Number of images within each mini-batch",)
parser.add_argument("--dropout", default=0.5, type = float)

class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def main(args):
    tensor = trfm.ToTensor()
    transforms = [tensor]
    args.dataset_root.mkdir(parents=True, exist_ok=True)

    train_dataset = UrbanSound8KDataset(‘UrbanSound8K_train.pkl’, mode)
    test_dataset = UrbanSound8KDataset(‘UrbanSound8K_test.pkl’, mode)

    train_loader = torch.utils.data.DataLoader( 
        train_dataset,
        batch_size=32,
        shuffle=True, 
        num_workers=8,
        pin_memory=True,
    ) 
     val_loader = torch.utils.data.DataLoader( 
        test_dataset,
        batch_size=32,
        shuffle=False, 
        num_workers=8,
        pin_memory=True,
    )

    #????????
    #           training code
    for i, (input, target, filename) in enumerate(val_loader):
    #           validation code
    #?????????

    model = CNN(height=85, width=41, channels=1, class_count=10, dropout=0.5)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), args.learning_rate, momentum=0.9)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )

    trainer = Trainer(
        model, train_loader, val_loader, criterion, optimizer, summary_writer, DEVICE
    )

    trainer.train(epochs=1, 2, 10, 10)

    summary_writer.close()


class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count: int, dropout: float):
        super().__init__()
        self.input_shape = ImageShape(height=height, width=width, channels=channels)
        self.class_count = class_count
        self.dropout = nn.Dropout(dropout)

        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=32,
            kernel_size=(3, 3),
            padding=(2, 2), #included?
            stride=(2, 2)
        )
        self.initialise_layer(self.conv1)
        self.bn32 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            padding=(2, 2),
            stride=(2, 2)
        )
        self.initialise_layer(self.conv2)
        #ALREADY DEFINED: self.bn32 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(2, 2),
            stride=(2, 2)
        )
        self.initialise_layer(self.conv3)
        self.bn64 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(2, 2),
            stride=(2, 2)
        )
        self.initialise_layer(self.conv4)
        #ALREADY DEFINED: self.bn64 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(15488, 1024)
        self.initialise_layer(self.fc1)

        self.fc2 = nn.Linear(1024, 10)
        self.initialise_layer(self.fc1)


    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn32(self.conv1(images)))
        x = F.relu(self.bn32(self.conv2(self.dropout(x))))
        x = self.pool1(x)
        x = F.relu(self.bn64(self.conv3(x)))
        x = F.relu(self.bn64(self.conv4(self.dropout(x))))
        x = torch.flatten(x,1)
        x = F.sigmoid(self.fc1(self.dropout(x))))
        x = F.softmax(self.fc2(x))
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0
    ):
        self.model.train()
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()

            for i, (input, target, filename) in enumerate(self.train_loader):
                batch = input.to(self.device)
                labels = target.to(self.device)
                data_load_end_time = time.time()

                logits = self.model.forward(batch)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                with torch.no_grad():
                    preds = logits.argmax(-1)
                    accuracy = compute_accuracy(labels, preds)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                self.validate()
                self.model.train()

    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "accuracy",
                {"train": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )

    def validate(self):
        results = {"preds": [], "labels": []}
        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for batch, labels in self.val_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=-1).cpu().numpy()
                results["preds"].extend(list(preds))
                results["labels"].extend(list(labels.cpu().numpy()))

        accuracy = compute_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )
        average_loss = total_loss / len(self.val_loader)

        pca = compute_pca(
            np.array(results["labels"]), np.array(results["preds"])
        )

        self.summary_writer.add_scalars(
                "accuracy",
                {"test": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )
        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")

        print(f"class 1 accuracy: {pca[0] * 100:2.2f}")
        print(f"class 2 accuracy: {pca[1] * 100:2.2f}")
        print(f"class 3 accuracy: {pca[2] * 100:2.2f}")
        print(f"class 4 accuracy: {pca[3] * 100:2.2f}")
        print(f"class 5 accuracy: {pca[4] * 100:2.2f}")
        print(f"class 6 accuracy: {pca[5] * 100:2.2f}")
        print(f"class 7 accuracy: {pca[6] * 100:2.2f}")
        print(f"class 8 accuracy: {pca[7] * 100:2.2f}")
        print(f"class 9 accuracy: {pca[8] * 100:2.2f}")
        print(f"class 10 t accuracy: {pca[9] * 100:2.2f}")


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
