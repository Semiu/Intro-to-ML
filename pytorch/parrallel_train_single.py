"""Script to train a model on a single device using PyTorch."""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# To take care of all possible devices and compatibility across boards
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits


class ToyDataset(Dataset):
    def __init__(self, X, y):
        """
        Sets up the data attributes that can be accessed using the class methods __getitem__ and __len__
        """
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        """
        Instruction to return exactly one item from the dataset using an index
        """
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        """
        To get the length of the dataset
        """
        return self.labels.shape[0]


if __name__ == "__main__":

    X_train = torch.tensor(
        [[-1.2, 3.1], [-0.9, 2.9], [-0.5, 2.6], [2.3, -1.1], [2.7, -1.5]]
    )
    y_train = torch.tensor([0, 0, 0, 1, 1])

    X_test = torch.tensor(
        [
            [-0.8, 2.8],
            [2.6, -1.6],
        ]
    )
    y_test = torch.tensor([0, 1])

    train_ds = ToyDataset(X_train, y_train)
    train_loader = DataLoader(
        dataset=train_ds, batch_size=2, shuffle=True, num_workers=0, drop_last=True
    )

    test_ds = ToyDataset(X_test, y_test)
    test_loader = DataLoader(
        dataset=test_ds, batch_size=2, shuffle=False, num_workers=0, drop_last=True
    )

    # instantiate the neural network model
    torch.manual_seed(123)
    model = NeuralNetwork(num_inputs=2, num_outputs=2)

    # New: Transfer the model onto the MPS
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    num_epochs = 3

    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, labels) in enumerate(train_loader):

            # New: Transfer the data onto the hardware.
            features, labels = features.to(device), labels.to(device)  # C
            logits = model(features)
            loss = F.cross_entropy(logits, labels)  # Loss function

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### LOGGING
            print(
                f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
                f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
                f" | Train/Val Loss: {loss:.2f}"
            )

        model.eval()
        # Optional model evaluation
