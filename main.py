import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class SimpleModel(nn.Module):
    # a simple MLP
    def __init__(
        self, n_input: int = 3072000, n_output: int = 3, n_h: int = 128, n_hl: int = 2
    ):
        super().__init__()

        self.n_input = n_input
        self.n_h = n_h
        self.n_hl = n_hl
        self.n_output = n_output

        self.input_layer = nn.Linear(self.n_input, self.n_h)
        self.output_layer = nn.Linear(self.n_h, self.n_output)
        self.hidden_layers = nn.ModuleList()
        for _ in range(self.n_hl - 1):
            self.hidden_layers.append(nn.Linear(self.n_h, self.n_h))

    def forward(self, x):
        h = F.relu(self.input_layer(x))
        for l in self.hidden_layers:
            h = F.relu(l(h))
        out = F.sigmoid(self.output_layer(h))
        return out


def img_processor(img_path):
    # a simple flattening processor
    img = cv2.imread(img_path)
    img = torch.FloatTensor(img).flatten().unsqueeze(0)
    return img


def labeler(filepath):
    id = filepath.split("/")[-1].split("_")[0]
    return id


def gen_labels(names):
    n_samples = len(names)
    n_unique_values = list(set(names))
    n_people = len(n_unique_values)
    n_output = n_people + 1  # add unidentified case
    matched_dict = {"unidentified": -1}
    for k, v in enumerate(n_unique_values):
        matched_dict[v] = k

    labels = torch.zeros((n_samples, n_output))
    for i, n in enumerate(names):
        id = matched_dict[n]
        labels[i, id] = 1
    return labels, matched_dict


def main():
    img_dirs = glob.glob("vein_dataset/train/**/*.jpg", recursive=True)
    data_x = []
    names = []
    for img_path in img_dirs:
        img = img_processor(img_path)
        label = labeler(img_path)

        data_x.append(img)
        names.append(label)
    data_x = torch.cat(data_x, dim=0)
    data_y, matched_dict = gen_labels(names)

    # Hyperparameters
    n_epochs = 100
    n_h = 256
    n_hl = 2
    batch_size = 4
    lr = 1e-3

    # Instance declarations
    model = SimpleModel(n_h=n_h, n_hl=n_hl).to(device)
    dataset = TensorDataset(data_x, data_y)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loops
    for i_epoch in range(n_epochs):
        for i_batch, batch in enumerate(dataloader):
            b_x, b_y = batch
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            b_y_hat = model(b_x)
            loss = F.mse_loss(b_y, b_y_hat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                f"Epoch {i_epoch+1} (Batch {i_batch+1}/{len(dataloader)}), Loss={loss.item():.5f}"
            )
            pass
    pass


if __name__ == "__main__":
    device = torch.device("mps")
    main()
