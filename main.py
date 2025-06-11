import glob
import inspect
import cv2
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from architectures.mlp import MLP
import time

# Enable Korean fonts in matplotlib on MacOS
rc("font", family="AppleGothic")
plt.rcParams["axes.unicode_minus"] = False


def timeit(func):
    def wrapper(*args, **kwargs):
        t1 = time.perf_counter()
        result = func(*args, **kwargs)
        t2 = time.perf_counter()
        print(f"Function \"{func.__name__}\" execution time: {t2-t1:.5f}s")
        return result

    return wrapper


def load_img(img_path: str):
    # load and tensorize
    img = cv2.imread(img_path)
    img = torch.FloatTensor(img).flatten().unsqueeze(0) / 255  # 1D flattened
    # img = torch.FloatTensor(img).unsqueeze(0) / 255 # 3D
    return img


def labeler(filepath: str):
    # filepath to id
    id = filepath.split("/")[-1].split("_")[0]
    return id


def gen_labels(names: list):
    n_samples = len(names)
    n_unique_values = list(set(names))
    n_people = len(n_unique_values)
    n_classes = n_people + 1  # add unidentified case
    matched_dict = {"unidentified": -1}
    for k, v in enumerate(n_unique_values):
        matched_dict[v] = k

    labels = torch.zeros((n_samples, n_classes))
    for i, n in enumerate(names):
        id = matched_dict[n]
        labels[i, id] = 1
    assert torch.equal(labels.sum(1), torch.ones(n_samples))  # Assure one-hot encoded
    return labels, matched_dict


@torch.no_grad()
def standardize_data(tensor: torch.Tensor):
    # (N,H,W,C) -> (1,H,W,C)
    std, mu = torch.std_mean(tensor, dim=0, keepdim=True)
    standardized = (tensor - mu) / std
    return standardized, mu, std


def save_model(model: nn.Module, matched_dict, destination: str):
    model_dict = {
        "state_dict": model.state_dict(),
        "matched_dict": matched_dict,
        "init_kwargs": model.init_kwargs,
    }
    print(f"Saved as {destination}")
    torch.save(model_dict, destination)


def load_model(path: str):
    model_dict = torch.load(path)
    model = MLP(**model_dict["init_kwargs"])
    model.load_state_dict(model_dict["state_dict"])
    model.eval()
    matched_dict = model_dict["matched_dict"]
    return model, matched_dict

@timeit
def train():
    # Load data
    img_dirs = glob.glob("vein_dataset/train/**/*.jpg", recursive=True)
    data_x = []
    names = []
    for img_path in img_dirs:
        img = load_img(img_path)
        label = labeler(img_path)

        data_x.append(img)
        names.append(label)
    data_x = torch.cat(data_x, dim=0)
    data_y, matched_dict = gen_labels(names)
    # data_x, mu_x, std_x = standardize_data(data_x)
    # data_y, mu_y, std_y = standardize_data(data_y)

    # Hyperparameters
    n_epochs = 1
    n_h = 128
    n_hl = 2
    batch_size = 4
    lr = 1e-5

    # Instance declarations
    model = MLP(3072000, 3, n_h=n_h, n_hl=n_hl).to(device)
    dataset = TensorDataset(data_x, data_y)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history_loss_epochsmean = np.full(n_epochs, 1.0)
    history_loss_iters = np.full(n_epochs * len(dataloader), 1.0)
    n_iter = 0
    # Training loops
    for i_epoch in range(n_epochs):
        bhistory_loss = np.full(len(dataloader), 1.0)
        for i_batch, batch in enumerate(dataloader):
            b_x, b_y = batch
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            b_y_hat = model(b_x)
            loss = F.cross_entropy(b_y, b_y_hat)

            for p in model.parameters():
                p.grad = None
            loss.backward()
            optimizer.step()

            bhistory_loss[i_batch] = loss.item()
            history_loss_iters[n_iter] = loss.item()
            n_iter += 1
            print(
                f"Epoch {i_epoch+1} (Batch {i_batch+1}/{len(dataloader)}), Loss={loss.item():.6f}"
            )
            pass
        history_loss_epochsmean[i_epoch] = np.mean(bhistory_loss)

    save_model(model, matched_dict, "model/test_mlp.pt")

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.suptitle("Loss by epochs")
    ax.plot(np.arange(n_epochs) + 1, history_loss_epochsmean)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("CELoss")
    ax.grid()
    fig.tight_layout()
    fig.savefig("loss_epochs.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.suptitle("Loss by iterations")
    ax.plot(np.arange(n_iter) + 1, history_loss_iters)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("CELoss")
    ax.grid()
    fig.tight_layout()
    fig.savefig("loss_iters.png", dpi=300)
    plt.close(fig)
    pass


@torch.no_grad()
def eval():
    model, matched_dict = load_model("model/test_mlp.pt")
    in_imgs = []
    out_probs = []
    # img_paths = glob.glob("vein_dataset/test/**/*.jpg")
    img_paths = glob.glob("250501/**/*.jpg")
    for img_path in img_paths:
        in_img = load_img(img_path)
        out_prob = model(in_img)

        in_imgs.append(in_img)
        out_probs.append(out_prob)

    for i, zipped in enumerate(zip(in_imgs, out_probs)):
        img_path, prob = zipped

        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        fig.suptitle(f"인식 이미지 경로: {img_paths[i]}")
        ax.imshow(img_path.reshape(800, 1280, 3))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

        annotation = "실제 신원: Person 1\n"
        annotation += f"인식 결과: Person {list(matched_dict.keys())[list(matched_dict.values()).index(torch.argmax(prob).item())]}\n"
        for k, v in matched_dict.items():
            annotation += f"신원 {k}: {prob[0,v]*100:.4f}%\n"
        annotation = annotation[:-1]  # drop last linebreak
        ax.text(
            0.7,
            0.1,
            annotation,
            transform=ax.transAxes,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )
        fig.tight_layout()
        fig.savefig(f"eval/250501_test_{i+1:02d}.png", dpi=300)
        plt.close(fig)
        pass


if __name__ == "__main__":
    device = torch.device("mps")
    train()
    eval()
