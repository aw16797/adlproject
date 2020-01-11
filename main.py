import time
import argparse
from pathlib import Path

from multiprocessing import cpu_count
from typing import Union, NamedTuple

import torch
import torchvision
import torchvision.datasets
import torch.backends.cudnn
import numpy as np
from torchvision.transforms import ToTensor
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import UrbanSound8KDataset

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Train CNN on UrbanSound8K dataset",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--mode", default="LMC", type=str, help="Which feature mode to execute for (MC, LMC or MLMC)")
parser.add_argument("--learning-rate", default=1e-3, type=float, help="Learning rate")
parser.add_argument("--dropout", default=0.5, type=float, help="Dropout variable")
parser.add_argument("--batch-size", default=32, type=int, help="Number of samples within each mini-batch")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs to train for")
parser.add_argument("--workers", default=8, type=int, help="Number of workers for loaders")
parser.add_argument("--decay", default=1e-3, type=float, help="Weight decay to use in SGD Optimizer")
parser.add_argument("--momentum", default=0.9, type=float, help="Learning rate")
parser.add_argument("--val-frequency", default=2, type=int, help="How frequently to test the model on the validation set in number of epochs")
parser.add_argument("--log-frequency", default=10, type=int, help="How frequently to save logs to tensorboard in number of steps")
parser.add_argument("--print-frequency", default=10, type=int, help="How frequently to print progress to the command line in number of steps")

class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def main(args):
    transform = ToTensor()

    train_data = UrbanSound8KDataset("UrbanSound8K_train.pkl", args.mode)
    test_data = UrbanSound8KDataset("UrbanSound8K_test.pkl", args.mode)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = CNN(height=85, width=41, channels=1, class_count=10, dropout=args.dropout)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.decay)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )

    trainer = Trainer(
        model, train_loader,  val_loader, criterion, optimizer, summary_writer, DEVICE
    )

    trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )

    summary_writer.close()


class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count: int, dropout: float):
        super().__init__()
        self.input_shape = ImageShape(height=height, width=width, channels=channels)
        self.class_count = class_count

        self.dropout = nn.Dropout(dropout)

        self.bn32 = nn.BatchNorm2d(32)
        self.bn64 = nn.BatchNorm2d(64)
        self.bnFC = nn.BatchNorm1d(1024)

        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=32,
            kernel_size=(3, 3),
            padding=(2, 2),
            stride=(2, 2),
        )
        self.initialise_layer(self.conv1)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32, # should be 64?
            kernel_size=(3, 3),
            padding=(2, 2),
            stride=(2, 2),
        )
        self.initialise_layer(self.conv2)

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(2, 2),
            stride=(2, 2),
        )
        self.initialise_layer(self.conv3)

        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(2, 2),
            stride=(2, 2),
        )
        self.initialise_layer(self.conv4)

        self.fc1 = nn.Linear(15488, 1024)
        self.initialise_layer(self.fc1)

        self.fc2 = nn.Linear(1024, 10)
        self.initialise_layer(self.fc2)


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn32(self.conv1(input)))

        x = F.relu(self.bn32(self.conv2(self.dropout(x))))

        x = self.pool(x)

        x = F.relu(self.bn64(self.conv3(x)))

        x = F.relu(self.bn64(self.conv4(self.dropout(x))))

        x = torch.flatten(x,1)

        # states dropout should in "2nd, 4th and Fully Connected layer"
        # should be in both FC layers or just first?
        # from lab 4:
            # "Dropout is typically applied on the input to FC layers,
            #  but can also be applied to the input of any hidden layer"
            # "In the forward method, apply self.dropout to the input of
            #  each fully connected layer."

        x = F.sigmoid(self.bnFC(self.fc1(self.dropout(x))))

        x = F.softmax(self.fc2(self.dropout(x)))

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
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()

    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )

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

    def validate(self, epoch, epochs):
        #results = {"preds": [], "labels": []}
        total_loss = 0
        self.model.eval()

        class_labels = []
        file_labels = []
        final_logits = torch.Tensor()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for i, (input, target, filename) in enumerate(val_loader):
                batch = input.to(self.device)
                labels = target.to(self.device)
                file_labels.append(filename)
                logits = self.model(batch)
                torch.cat(final_logits, logits, dim=0)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                #preds = logits.argmax(dim=-1).cpu().numpy()
                #results["preds"].extend(list(preds))
                #results["labels"].extend(list(labels.cpu().numpy()))
                class_labels.append(list(labels.cpu().numpy()))

        # save logits to file
        if (epoch == epochs-1):
            torch.save(final_logits, 'lmc.pt')
            torch.save(torch.ToTensor(files), 'files.pt')

        average_loss = total_loss / len(self.val_loader)

        #softmax to scores
        final_scores = torch.Tensor()
        logits_length = final_logits.size()
        for i in range (0, logits_length[0]):
            scores = torch.tensor(F.softmax(final_logits[i, :]))
            if( i == 0):
                final_scores = torch.cat([final_scores, scores], dim=0)
            else:
                final_scores = torch.stack([final_scores, scores], dim=0)

        pca = compute_pca(
            class_labels, file_labels, final_scores
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
        print(f"class 10 accuracy: {pca[9] * 100:2.2f}")

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
    class_labels: Union[torch.Tensor, np.ndarray],
    file_labels: Union[torch.Tensor, np.ndarray],
    scores: Union[torch.Tensor, np.ndarray],
):

    file_label_dict = {}    #to store correct class label of each file
    file_count_dict = {}    #to store number of segments relating to each file
    file_score_dict = {}    #to store scores of each segment to related file

    scores_size = scores.size()
    for i in range (0, scores_size[0]):
        x = file_labels[i]                        # x = file of segment with scores[i]
        file_label_dict[x] = class_labels[i]      #save actual label for file[x] in dictionary
        if x in file_score_dict:
            file_count_dict[x] += 1
            file_score_dict[x] += scores[i]
        else:
            file_count_dict[x] = 1
            file_score_dict[x] = scores[i]

    file_avg_dict = {}      #to store average score for each file
    file_pred_dict = {}     #to store class prediction for each file

    for key, val in file_score_dict.items():
        file_avg_dict[key] = val/file_count_dict[key]
        file_pred_dict[key] = file_avg_dict[key].argmax(dim=-1).cpu().numpy()

    #Number of files for each class
    total_class_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

    #Correctly predicted classes
    correct_class_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

    #PCA
    pca_dict = {}

    for key, val in file_label_dict.items():          #for all files
        total_class_dict[val] += 1                    #count number of files for each class
        if(val == file_pred_dict[key]):               #if file is correctly predicted...
            correct_class_dict[val] += 1              #count correct prediction of file to class

    for key, val in total_class_dict.items():         #calculate pca
        if(total_class_dict[key] != 0):
          pca_dict[key] = (correct_class_dict[key]/total_class_dict[key])
        else:
          pca_dict[key] = 0

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
