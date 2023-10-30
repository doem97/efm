from pathlib import Path
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import pdb

# import wandb


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


class CombinedDatasetMy(Dataset):
    def __init__(self, imagenet_dataset, my_dataset, N, num_pairs=4):
        self.imagenet_dataset = imagenet_dataset
        self.my_dataset = my_dataset
        self.N = N
        self.num_pairs = num_pairs

    def __len__(self):
        return min(len(self.imagenet_dataset), len(self.my_dataset))

    def __getitem__(self, idx):
        imagenet_image, _ = self.imagenet_dataset[idx]
        my_image, my_label = self.my_dataset[idx]

        imagenet_image, perms = permuteNxN(
            imagenet_image, self.N, num_pairs=self.num_pairs
        )
        y_in = perm2vecmatNxN(perms, self.N)

        return {
            "imagenet_image": imagenet_image,
            "y_in": y_in,
            "perms": perms,
            "my_image": my_image,
            "my_label": my_label,
        }


class CombinedDatasetIN(Dataset):
    def __init__(self, imagenet_dataset, my_dataset, N, num_pairs=4):
        self.imagenet_dataset = imagenet_dataset
        self.my_dataset = my_dataset
        self.N = N
        self.num_pairs = num_pairs

    def __len__(self):
        return len(self.imagenet_dataset)

    def __getitem__(self, idx):
        imagenet_image, _ = self.imagenet_dataset[idx]

        # If idx is larger than the length of my_dataset, loop back to the start
        # my_idx = idx % len(self.my_dataset)
        # my_image, my_label = self.my_dataset[my_idx]

        imagenet_image, perms = permuteNxN(
            imagenet_image, self.N, num_pairs=self.num_pairs
        )
        y_in = perm2vecmatNxN(perms, self.N)

        return {
            "imagenet_image": imagenet_image,
            "y_in": y_in,
            "perms": perms,
            # "my_image": my_image,
            # "my_label": my_label,
        }


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


class SimpleConvNet(nn.Module):
    def __init__(self, patch_size=56):
        super(SimpleConvNet, self).__init__()
        # 3 x patch_size x patch_size input
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

        # Calculate the size of the feature map after the convolution and pooling layers
        def convpool_out_size(
            size, kernel_size=3, stride=1, padding=1, pool_kernel_size=2, pool_stride=2
        ):
            # Convolution layer
            size = (size - kernel_size + 2 * padding) // stride + 1
            # Pooling layer
            size = (size - pool_kernel_size) // pool_stride + 1
            return size

        # Apply the convpool_out_size function three times since there are three sets of convolution and pooling layers
        final_map_size = convpool_out_size(
            convpool_out_size(convpool_out_size(patch_size))
        )

        # Calculate the total number of features after the final pooling layer
        self.num_flat_features = 256 * final_map_size * final_map_size

        self.fc1 = nn.Linear(self.num_flat_features, 1024)
        self.fc2 = nn.Linear(1024, 768)
        self.dropout = nn.Dropout(0.5)
        self.bn4 = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(-1, self.num_flat_features)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class JigsawNet(nn.Module):
    """
    A neural network that solves 6x6 jigsaw puzzles of 56x56 patches.
    """

    def __init__(self, sinkhorn_iter=0, N=6, patch_size=56):
        super().__init__()
        self.num_classes = 50
        self.N = N
        self.patch_num = N * N
        self.patch_size = patch_size
        self.conv_net = SimpleConvNet(patch_size=patch_size)
        self.embedding_dim = 768  # Dimension of the embedding for each patch

        # 768-dimensional embedding for each of the 36 patches
        self.fc1 = nn.Linear(self.embedding_dim * self.patch_num, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, self.patch_num * self.patch_num)
        self.sinkhorn_iter = sinkhorn_iter

        # Calculate the input dimension for the classifier
        classifier_input_dim = self.embedding_dim * patch_num
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, self.num_classes),
        )

    def embed(self, x):
        bs, c, h, w = x.size()
        patch_size = self.patch_size
        patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(
            bs, c, self.N, self.N, patch_size, patch_size
        )
        patches = (
            patches.permute(2, 3, 0, 1, 4, 5)
            .contiguous()
            .view(-1, c, patch_size, patch_size)
        )

        # Embed each patch
        embeddings = self.conv_net(patches)
        embeddings = (
            embeddings.view(self.N, self.N, bs, -1).permute(2, 0, 1, 3).contiguous()
        )
        embeddings = embeddings.view(embeddings.size(0), -1)
        return embeddings

    # def forward(self, x1, x2):
    #     # x1 is imneet images
    #     # x2 is my images
    #     bs, c, h, w = x1.size()
    #     embeddings1 = self.embed(x1)

    #     # Permutation prediction
    #     x1 = F.dropout(embeddings1, p=0.1, training=self.training)
    #     x1 = F.relu(self.fc1_bn(self.fc1(x1)))
    #     x1 = torch.sigmoid(self.fc2(x1))

    #     if self.sinkhorn_iter > 0:
    #         x1 = x1.view(bs, self.patch_num, self.patch_num)
    #         x1 = sinkhorn(x1, self.sinkhorn_iter)
    #         # x1 dim: bs x patch_num x patch_num
    #         x1 = x1.view(bs, -1)

    #     # Classification
    #     embeddings2 = self.embed(x2)
    #     x2 = self.classifier(embeddings2)
    #     return x1, x2

    def forward(self, x1):
        # x1 is imneet images
        # x2 is my images
        bs, c, h, w = x1.size()
        embeddings1 = self.embed(x1)

        # Permutation prediction
        x1 = F.dropout(embeddings1, p=0.1, training=self.training)
        x1 = F.relu(self.fc1_bn(self.fc1(x1)))
        x1 = torch.sigmoid(self.fc2(x1))

        if self.sinkhorn_iter > 0:
            x1 = x1.view(bs, self.patch_num, self.patch_num)
            x1 = sinkhorn(x1, self.sinkhorn_iter)
            # x1 dim: bs x patch_num x patch_num
            x1 = x1.view(bs, -1)

        # Classification
        # embeddings2 = self.embed(x2)
        # x2 = self.classifier(embeddings2)
        return x1


# def permuteNxN(image, n):
#     """
#     Splits the image into n x n patches and randomly permutes the patches.
#     Assumes image is a single image tensor of shape [channels, height, width].
#     """
#     assert (
#         image.size(1) % n == 0 and image.size(2) % n == 0
#     ), "Image dimensions must be divisible by n"
#     patch_size = image.size(1) // n

#     p_image = torch.FloatTensor(image.size())
#     perm = torch.randperm(n * n)
#     for j in range(n * n):
#         sr, sc = divmod(j, n)
#         tr, tc = divmod(perm[j].item(), n)
#         p_image[
#             :,
#             tr * patch_size : (tr + 1) * patch_size,
#             tc * patch_size : (tc + 1) * patch_size,
#         ] = image[
#             :,
#             sr * patch_size : (sr + 1) * patch_size,
#             sc * patch_size : (sc + 1) * patch_size,
#         ]
#     return p_image, perm


def permuteNxN(image, n, num_pairs=4):
    """
    Splits the image into n x n patches and randomly permutes the specified number of patch pairs.
    Assumes image is a single image tensor of shape [channels, height, width].

    :param image: Input image tensor of shape [channels, height, width].
    :param n: The image will be divided into n x n patches.
    :param num_pairs: Number of patch pairs to be permuted. If None, all patches are permuted.
    :return: A tuple of (permuted_image, permutation_indices).
    """
    assert (
        image.size(1) % n == 0 and image.size(2) % n == 0
    ), "Image dimensions must be divisible by n"
    patch_size = image.size(1) // n

    p_image = torch.FloatTensor(image.size())
    total_patches = n * n
    perm = torch.arange(total_patches)
    _perm = torch.arange(total_patches)

    if num_pairs is not None:
        assert (
            0 <= num_pairs <= total_patches // 2
        ), "num_pairs must be between 0 and total_patches // 2"
        selected_indices = torch.randperm(total_patches)[: 2 * num_pairs]
        for i in range(0, len(selected_indices), 2):
            idx1, idx2 = selected_indices[i], selected_indices[i + 1]
            perm[idx1], perm[idx2] = perm[idx2], _perm[idx1]

    for j in range(total_patches):
        sr, sc = divmod(j, n)
        tr, tc = divmod(perm[j].item(), n)
        p_image[
            :,
            tr * patch_size : (tr + 1) * patch_size,
            tc * patch_size : (tc + 1) * patch_size,
        ] = image[
            :,
            sr * patch_size : (sr + 1) * patch_size,
            sc * patch_size : (sc + 1) * patch_size,
        ]

    return p_image, perm


# # Example usage:
# image = torch.rand(3, 128, 128)  # [channels, height, width]
# n = 4  # Split into 4x4 patches
# num_pairs = 2  # Permute 2 pairs of patches
# p_image, perm = permuteNxN(image, n, num_pairs)
# print("Permuted Indices:", perm)


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


def perm2vecmatNxN(perm, n):
    """
    Converts a permutation vector to a vectorized assignment matrix.
    Assumes perm is a single permutation vector of length n*n.
    """
    mat = torch.zeros(n * n, n * n)
    for k in range(n * n):
        mat[k, perm[k]] = 1.0
    return mat.view(-1)


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
def compute_acc(p_pred, p_true, N, average=False):
    """
    We require that the location of all four pieces are correctly predicted.
    Note: this function is compatible with GPU tensors.
    """
    # Remember to cast to float.
    # How many images are correctly predicted
    n = torch.sum((torch.sum(p_pred == p_true, 1) == N**2).float())
    if average:
        return n / p_pred.size()[0]
    else:
        return n


def format_time(seconds):
    """Formats time from seconds to hh:mm:ss"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def train_model(
    model: nn.Module,
    perm_criterion: nn.Module,
    my_criterion: nn.Module,
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
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(n_epochs):
        avg_loss, acc, n_samples = run_epoch(
            model,
            train_loader,
            perm_criterion,
            my_criterion,
            optimizer,
            device,
            N,
            is_train=True,
            epoch=epoch,
            total_epochs=n_epochs,
        )
        if scheduler is not None:
            scheduler.step()
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(acc)

        # Clear GPU memory cache
        if device == "cuda":
            torch.cuda.empty_cache()

        avg_loss, acc, n_samples = run_epoch(
            model,
            validation_loader,
            perm_criterion,
            my_criterion,
            optimizer,
            device,
            N,
            is_train=False,
            print_interval=20,
        )
        history["val_loss"].append(avg_loss)
        history["val_acc"].append(acc)

        if dist.get_rank() == 0:
            print(
                f"Epoch {epoch + 1:03d}: "
                f"loss={history['train_loss'][-1]:.4f}, "
                f"val_loss={history['val_loss'][-1]:.4f}, "
                f"acc={history['train_acc'][-1]:.2%}, "
                f"val_acc={history['val_acc'][-1]:.2%}"
            )
        if checkpoint_name is not None and torch.distributed.get_rank() == 0:
            torch.save(
                {
                    "history": history,
                    "model": model.module.state_dict(),  # Save the original model's state dict
                    "optimizer": optimizer.state_dict(),
                },
                f"{checkpoint_name}_{epoch}.pth",
            )

    if dist.get_rank() == 0:
        print("Training completed")

    return history


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    imagenet_criterion: nn.Module,
    my_data_criterion: nn.Module,
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

    for i, data in enumerate(data_loader, start=1):
        # imagenet_images, y_in, my_images, my_labels = (
        #     data["imagenet_image"],
        #     data["y_in"],
        #     data["my_image"],
        #     data["my_label"],
        # )
        # imagenet_images, y_in, my_images, my_labels = (
        #     imagenet_images.to(device),
        #     y_in.to(device),
        #     my_images.to(device),
        #     my_labels.to(device),
        # )
        imagenet_images, y_in = (
            data["imagenet_image"],
            data["y_in"],
        )
        imagenet_images, y_in = (
            imagenet_images.to(device),
            y_in.to(device),
        )
        batch_size = imagenet_images.size(0)

        if is_train:
            optimizer.zero_grad()

        # imagenet_output, my_output = model(imagenet_images, my_images)

        imagenet_output = model(imagenet_images)
        imagenet_loss = imagenet_criterion(imagenet_output, y_in)
        loss = imagenet_loss

        # my_loss = my_data_criterion(my_output, my_labels)
        # loss += my_loss  # Combine the losses from the two datasets

        if is_train:
            loss.backward()
            optimizer.step()

        # Calculate accuracy for your dataset's classification
        # _, predicted = torch.max(my_output.data, 1)
        # n_correct_pred += (predicted == my_labels).sum().item()

        # Calculate accuracy for imagenet dataset's permutation prediction using compute_acc
        # The imagenet_output is of size [batch_size, patch_num, patch_num]
        # y_in is of size [batch_size, patch_num, patch_num]
        # imagenet_output dim: (bs, patch_num x patch_num)
        y_in = vecmat2permNxN(y_in, N)
        # y_in dim now: (bs, patch_num)
        imagenet_output = vecmat2permNxN(imagenet_output, N)
        n_correct_pred += compute_acc(imagenet_output, y_in, N, average=False)
        n_samples += batch_size

        running_loss += loss.item() * batch_size

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
            # Log to wandb or any other logging tool you prefer
            # Make sure you only log from the main process if using distributed training
            # if dist.get_rank() == 0:
            # wandb.log(
            #     {
            #         "avg_loss": avg_loss,
            #         "avg_acc": avg_acc,
            #         "lr": optimizer.param_groups[0]["lr"],
            #     }
            # )

    return running_loss / n_samples, n_correct_pred / n_samples, n_samples


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    torch.manual_seed(42)

    N = 2
    num_pairs = 2
    patch_num = N * N
    image_size = 336
    patch_size = image_size // N

    batch_size = 128

    assert image_size == patch_size * N
    dataset_dir = Path("./data/imagenet/ILSVRC/Data/CLS-LOC")
    my_dataset_dir = Path("./data/cspuzzle")
    output_dir = Path(f"./outputs/in1k_{image_size}_{patch_size}_{N}")
    output_dir.mkdir(parents=True, exist_ok=True)
    n_epochs = 100
    sinkhorn_iter = 5
    weight_decay = 5e-4

    transform1 = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    transform2 = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = datasets.ImageFolder(
        root=dataset_dir / "train", transform=transform1
    )
    my_train_dataset = datasets.ImageFolder(
        root=my_dataset_dir / "train", transform=transform2
    )
    combined_train_dataset = CombinedDatasetIN(
        train_dataset, my_train_dataset, N, num_pairs=num_pairs
    )
    combined_train_sampler = DistributedSampler(combined_train_dataset)
    combined_train_loader = DataLoader(
        combined_train_dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        sampler=combined_train_sampler,
    )

    val_dataset = datasets.ImageFolder(root=dataset_dir / "val", transform=transform1)
    my_val_dataset = datasets.ImageFolder(
        root=my_dataset_dir / "val", transform=transform2
    )
    combined_val_dataset = CombinedDatasetIN(
        val_dataset, my_val_dataset, N, num_pairs=num_pairs
    )
    combined_val_sampler = DistributedSampler(combined_val_dataset)
    combined_val_loader = DataLoader(
        combined_val_dataset,
        batch_size=32,
        num_workers=0,
        pin_memory=True,
        sampler=combined_val_sampler,
    )

    model = JigsawNet(sinkhorn_iter=sinkhorn_iter, N=N, patch_size=patch_size).to(
        device
    )
    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    n_params = sum(np.prod(p.size()) for p in model.parameters())
    if local_rank == 0:
        print(f"# of parameters: {n_params}")

    perm_criterion = nn.BCELoss().to(device)
    my_criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    checkpoint_name = output_dir / f"e{n_epochs}_s{sinkhorn_iter}"
    history = train_model(
        model,
        perm_criterion,
        my_criterion,
        optimizer,
        combined_train_loader,
        combined_val_loader,
        n_epochs=n_epochs,
        checkpoint_name=checkpoint_name,
        N=N,
        scheduler=scheduler,
    )
