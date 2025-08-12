"""Script to parrallel-train a model, with compatibility with CUDA, MPS and CPU, using PyTorch."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import platform
import os
from torch.utils.data.distributed import DistributedSampler # to split the dataset across multiple processes
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, is_initialized, get_rank # respectively to initialize and destroy the process group in the distributed training modes


from parrallel_train_single import NeuralNetwork, ToyDataset, device as DEVICE


def ddp_setup(rank: int, world_size: int):
    """
    Setup the distributed environment for training. (1 process/gpu)
    This function initializes the process group and sets the device for each process.
    It should be called before any other distributed operations.
    This function sets the environment variables for the master address and port,
    which are used to establish communication between processes.
    It also sets the device for each process based on its rank.
    This is necessary for distributed training to work correctly.
    Arguments:
        rank: a unique process ID
        world_size: total number of processes in the group
    """
    # Only set MASTER_ADDR and MASTER_PORT if not already defined by torchrun
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"

    # Choose backend
    if torch.device == "cuda" and platform.system() != "Windows":
        backend = "nccl"   # CUDA only, not on Windows
    else:
        backend = "gloo"   # Works on CPU, MPS, Windows

    # Initialize the process group
    init_process_group(backend=backend, rank=rank, world_size=world_size)

    # Prefer LOCAL_RANK if launched with torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if torch.device == "cuda":
        # Set the device for each process based on its rank
        # This ensures that each process uses a different GPU
        # rank: unique process ID, world_size: total number of processes in the group
        num = torch.cuda.device_count()
        if num == 0:
            raise RuntimeError("No GPUs available for distributed training.")
        idx = local_rank % num
        torch.cuda.set_device(idx)
    elif torch.device == "mps":
        # For MPS, we can use the same device for all processes
        if hasattr(torch, "mps") and hasattr(torch.mps, "set_device"):
            torch.mps.set_device(0)
    else:
        # For CPU, we can use the same device for all processes
        pass


def prepare_dataset():
    """
    Toy dataset for testing the distributed training script.
    """
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])

    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])
    y_test = torch.tensor([0, 1])

    # Uncomment these lines to increase the dataset size to run this script on up to 8 GPUs:
    # factor = 4
    # X_train = torch.cat([X_train + torch.randn_like(X_train) * 0.1 for _ in range(factor)])
    # y_train = y_train.repeat(factor)
    # X_test = torch.cat([X_test + torch.randn_like(X_test) * 0.1 for _ in range(factor)])
    # y_test = y_test.repeat(factor)

    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=False,  # NEW: False because of DistributedSampler below
        pin_memory=True,
        drop_last=True,
        # NEW: chunk batches across GPUs without overlapping samples:
        sampler=DistributedSampler(train_ds)  # NEW
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
    )
    return train_loader, test_loader


def main(rank, world_size, num_epochs):
    """
    Main wrapper
    """

    ddp_setup(rank, world_size)  # NEW: initialize process groups

    train_loader, test_loader = prepare_dataset()

    if DEVICE.type == "cuda":
        this_device = torch.device(f"cuda:{torch.cuda.current_device()}")
    elif DEVICE.type == "mps":
        this_device = torch.device("mps")
    else:
        this_device = torch.device("cpu")


    model = NeuralNetwork(num_inputs=2, num_outputs=2).to(this_device)  # NEW: move model to device

    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    # Wrap with DDP only when we are actually distributed
    if world_size > 1:
        if DEVICE.type == "cuda":
            model = DDP(
                model,
                device_ids=[torch.cuda.current_device()],
                output_device=torch.cuda.current_device(),
                gradient_as_bucket_view=True
            )
        else:
            # CPU/MPS: don't pass device_ids
            model = DDP(model, gradient_as_bucket_view=True)

    for epoch in range(num_epochs):
        # NEW: Set sampler to ensure each epoch has a different shuffle order
        train_loader.sampler.set_epoch(epoch)

        model.train()
        for features, labels in train_loader:

            features, labels = features.to(this_device), labels.to(this_device) 
            logits = model(features)
            loss = F.cross_entropy(logits, labels)  # Loss function

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # LOGGING
            if world_size == 1 or (is_initialized() and get_rank() == 0):
                print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batchsize {labels.shape[0]:03d}"
                  f" | Train/Val Loss: {loss:.2f}")
    # EVALUATION
    try:
        train_acc = compute_accuracy(model, train_loader, device=this_device)
        test_acc = compute_accuracy(model, test_loader, device=this_device)
        if world_size == 1 or (is_initialized() and get_rank() == 0):
            print(f"Train Accuracy: {train_acc:.2f}")
            print(f"Test Accuracy: {test_acc:.2f}")
   
    except ZeroDivisionError as e:
        raise ZeroDivisionError(
            f"{e}\n\nThis script is designed for 2 GPUs. You can run it as:\n"
            "torchrun --nproc_per_node=2 DDP-script-torchrun.py\n"
            f"Or, to run it on {torch.cuda.device_count()} GPUs, uncomment the code on lines 103 to 107."
        )


    if is_initialized():
        destroy_process_group()  # NEW: cleanly exit distributed mode


def compute_accuracy(model, dataloader, device):
    model.eval()
    correct = 0.0
    total_examples = 0

    for _, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)
    return (correct / total_examples).item()


if __name__ == "__main__":
    # NEW: Use environment variables set by torchrun if available, otherwise default to single-process.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))

    if rank == 0:
        print("PyTorch:", torch.__version__)
        print("Device:", DEVICE.type)
        if DEVICE.type == "cuda":
            print("CUDA GPUs:", torch.cuda.device_count())

    torch.manual_seed(123) # For reproducibility
    num_epochs = 3
    main(rank, world_size, num_epochs)