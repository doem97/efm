from pathlib import Path
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler


def load_model(
    checkpoint_path: str, model: nn.Module, optimizer: optim.Optimizer = None
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer


def sinkhorn(A, n_iter=4):
    """
    Sinkhorn iterations.
    """
    for _ in range(n_iter):
        A = A / A.sum(dim=1, keepdim=True)
        A = A / A.sum(dim=2, keepdim=True)
    return A


class SimpleConvNet16(nn.Module):
    """
    A simple convolutional neural network shared among all pieces.
    """

    def __init__(self):
        super().__init__()
        # 3 x 16 x 16 input
        self.conv1 = nn.Conv2d(3, 8, 3)
        # 8 x 14 x 14
        self.conv2 = nn.Conv2d(8, 8, 3)
        self.conv2_bn = nn.BatchNorm2d(8)
        # 8 x 12 x 12
        self.pool1 = nn.MaxPool2d(2, 2)
        # 8 x 6 x 6
        self.conv3 = nn.Conv2d(8, 16, 3)
        self.conv3_bn = nn.BatchNorm2d(16)
        # 16 x 4 x 4
        self.fc1 = nn.Linear(16 * 4 * 4, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        # 128-d features
        self.fc2 = nn.Linear(128, 128)
        self.fc2_bn = nn.BatchNorm1d(128)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        return x


class SimpleConvNet56(nn.Module):
    def __init__(self):
        super(SimpleConvNet56, self).__init__()
        # 3 x 56 x 56 input
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(256 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 768)
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(-1, 256 * 7 * 7)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class JigsawNet(nn.Module):
    """
    A neural network that solves 6x6 jigsaw puzzles of 56x56 patches.
    """

    def __init__(self, sinkhorn_iter=0):
        super().__init__()
        self.conv_net = SimpleConvNet56()
        # 768-dimensional embedding for each of the 36 patches
        self.fc1 = nn.Linear(768 * 36, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 36 * 36)  # 36 x 36 assignment matrix
        self.sinkhorn_iter = sinkhorn_iter

    def forward(self, x):
        bs, c, h, w = x.size()
        patch_size = 56
        patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(bs, c, 6, 6, patch_size, patch_size)
        patches = (
            patches.permute(2, 3, 0, 1, 4, 5)
            .contiguous()
            .view(-1, c, patch_size, patch_size)
        )

        # Embed each patch
        embeddings = self.conv_net(patches)
        embeddings = (
            embeddings.view(6, 6, bs, -1).permute(2, 0, 1, 3).contiguous().view(bs, -1)
        )

        # Dense layers
        x = F.dropout(embeddings, p=0.1, training=self.training)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))

        if self.sinkhorn_iter > 0:
            x = x.view(bs, 36, 36)
            x = sinkhorn(x, self.sinkhorn_iter)
            x = x.view(bs, -1)

        return x


def permuteNxN(images, n):
    """
    Splits the images into n x n patches and randomly permutes the patches.
    """
    assert (
        images.size(2) % n == 0 and images.size(3) % n == 0
    ), "Image dimensions must be divisible by n"
    patch_size = images.size(2) // n

    p_images = torch.FloatTensor(images.size())
    perms = torch.LongTensor(images.size(0), n * n)
    for i in range(images.size(0)):
        p = torch.randperm(n * n)
        for j in range(n * n):
            sr, sc = divmod(j, n)
            tr, tc = divmod(p[j].item(), n)
            p_images[
                i,
                :,
                tr * patch_size : (tr + 1) * patch_size,
                tc * patch_size : (tc + 1) * patch_size,
            ] = images[
                i,
                :,
                sr * patch_size : (sr + 1) * patch_size,
                sc * patch_size : (sc + 1) * patch_size,
            ]
        perms[i, :] = p
    return (p_images, perms)


def restoreNxN(p_images, perms, n):
    """
    Restores the original image from the patches and the given permutation.
    """
    assert (
        p_images.size(2) % n == 0 and p_images.size(3) % n == 0
    ), "Image dimensions must be divisible by n"
    patch_size = p_images.size(2) // n

    images = torch.FloatTensor(p_images.size())
    for i in range(images.size(0)):
        for j in range(n * n):
            sr, sc = divmod(j, n)
            tr, tc = divmod(perms[i, j].item(), n)
            images[
                i,
                :,
                sr * patch_size : (sr + 1) * patch_size,
                sc * patch_size : (sc + 1) * patch_size,
            ] = p_images[
                i,
                :,
                tr * patch_size : (tr + 1) * patch_size,
                tc * patch_size : (tc + 1) * patch_size,
            ]
    return images


def perm2vecmatNxN(perms, n):
    """
    Converts permutation vectors to vectorized assignment matrices.
    """
    n_samples = perms.size(0)
    mat = torch.zeros(n_samples, n * n, n * n)
    for i in range(n_samples):
        for k in range(n * n):
            mat[i, k, perms[i, k]] = 1.0
    return mat.view(n_samples, -1)


def vecmat2permNxN(x, n):
    """
    Converts vectorized assignment matrices back to permutation vectors.
    """
    n_samples = x.size(0)
    x = x.view(n_samples, n * n, n * n)
    _, ind = x.max(2)
    return ind


def imshow(img, title=None):
    """
    Displays a torch image.
    """
    img_mean = 0.5
    img_std = 0.5
    img = img * img_std + img_mean
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title is not None:
        plt.title(title)


# Test helper
def compute_acc(p_pred, p_true, N, average=True):
    """
    We require that the location of all four pieces are correctly predicted.
    Note: this function is compatible with GPU tensors.
    """
    # Remember to cast to float.
    n = torch.sum((torch.sum(p_pred == p_true, 1) == N).float())
    if average:
        return n / p_pred.size()[0]
    else:
        return n


def train_model(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    n_epochs: int = 40,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    checkpoint_name: Path = None,
    N: int = 3,
    scheduler: lr_scheduler._LRScheduler = None,
) -> dict:
    model.to(device)
    history = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(n_epochs):
        running_loss, n_correct_pred, n_samples = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            N,
            is_train=True,
            epoch=epoch,
            total_epochs=n_epochs,
        )
        history["loss"].append(running_loss)
        history["acc"].append(n_correct_pred / n_samples)

        running_loss, n_correct_pred, n_samples = run_epoch(
            model, validation_loader, criterion, optimizer, device, N, is_train=False
        )
        history["val_loss"].append(running_loss)
        history["val_acc"].append(n_correct_pred / n_samples)

        if dist.get_rank() == 0:
            print(
                f"Epoch {epoch + 1:03d}: "
                f"loss={history['loss'][-1]:.4f}, "
                f"val_loss={history['val_loss'][-1]:.4f}, "
                f"acc={history['acc'][-1]:.2%}, "
                f"val_acc={history['val_acc'][-1]:.2%}"
            )

        if scheduler is not None:
            scheduler.step()

    if dist.get_rank() == 0:
        print("Training completed")

    if checkpoint_name is not None and torch.distributed.get_rank() == 0:
        torch.save(
            {
                "history": history,
                "model": model.module.state_dict(),  # Save the original model's state dict
                "optimizer": optimizer.state_dict(),
            },
            checkpoint_name,
        )

    return history


def format_time(seconds):
    """Formats time from seconds to hh:mm:ss"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    N: int,
    is_train: bool,
    print_interval: int = 20,  # print every 20 batches
    epoch: int = 0,
    total_epochs: int = 0,
) -> tuple:
    if is_train:
        model.train()
        phase = "Training"
    else:
        model.eval()
        phase = "Validation"

    running_loss = 0.0
    n_correct_pred = 0
    n_samples = 0
    start_time = time.time()

    for i, (inputs, _) in enumerate(data_loader, start=1):
        inputs, perms = permuteNxN(inputs, N)
        y_in = perm2vecmatNxN(perms, N)
        inputs, y_in, perms = inputs.to(device), y_in.to(device), perms.to(device)

        if is_train:
            optimizer.zero_grad()

        outputs = model(inputs)
        acc = compute_acc(vecmat2permNxN(outputs, N), perms, N * N, True)
        loss = criterion(outputs, y_in)

        if is_train:
            loss.backward()
            optimizer.step()

        n_samples += inputs.size(0)
        n_correct_pred += acc.item() * inputs.size(0)
        running_loss += loss.item() * inputs.size(0)

        if i % print_interval == 0 and dist.get_rank() == 0:
            elapsed_time = time.time() - start_time
            avg_loss = running_loss / n_samples
            avg_acc = n_correct_pred / n_samples
            remain_batches = len(data_loader) - i
            remain_time = elapsed_time / i * remain_batches
            print(
                f"{phase} Epoch [{epoch+1}/{total_epochs}], "
                f"Step [{i}/{len(data_loader)}], "
                f"Loss: {avg_loss:.4f}, "
                f"Accuracy: {avg_acc * 100:.2f}%, "
                f"Elapsed Time: {format_time(elapsed_time)}, "
                f"Remaining Time: {format_time(remain_time)}"
            )

    return running_loss / n_samples, n_correct_pred / n_samples, n_samples


def test_model(model: nn.Module, test_loader: DataLoader, N: int) -> float:
    _, n_correct_pred, n_samples = run_epoch(
        model,
        test_loader,
        None,
        None,
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        N,
        is_train=False,
    )
    acc = n_correct_pred / n_samples
    return acc


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    torch.manual_seed(42)

    N = 6
    batch_size = 128
    dataset_dir = Path("./data/imagenet/ILSVRC/Data/CLS-LOC")
    output_dir = Path("./outputs/jigsaw_in1k_336_56")
    n_epochs = 100
    sinkhorn_iter = 5
    weight_decay = 1e-4

    transform = transforms.Compose(
        [
            transforms.Resize((336, 336)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = datasets.ImageFolder(
        root=dataset_dir / "train", transform=transform
    )
    subset_indices = np.random.choice(
        len(train_dataset), size=int(0.5 * len(train_dataset)), replace=False
    )
    train_dataset = Subset(train_dataset, subset_indices)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=10,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_dataset = datasets.ImageFolder(root=dataset_dir / "val", transform=transform)
    val_sampler = DistributedSampler(val_dataset)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=10,
        pin_memory=True,
        sampler=val_sampler,
    )

    model = JigsawNet(sinkhorn_iter=sinkhorn_iter).to(device)
    model = DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    )

    n_params = sum(np.prod(p.size()) for p in model.parameters())
    if local_rank == 0:
        print(f"# of parameters: {n_params}")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    checkpoint_name = dataset_dir / f"e{n_epochs}_s{sinkhorn_iter}.pk"
    history = train_model(
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        n_epochs=n_epochs,
        checkpoint_name=checkpoint_name,
        N=N,
        scheduler=scheduler,
    )

    print(f"Training accuracy: {test_model(model, train_loader, N)}")
    print(f"Validation accuracy: {test_model(model, val_loader, N)}")
