import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import get_dataset
class BrainCNN(nn.Module):

    def __init__(self):
        super(BrainCNN, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(

            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 2)
        )

    def forward(self, x):

        x = self.conv(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    dataset = get_dataset()

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0   
    )

    print("Total samples:", len(dataset))
    model = BrainCNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )
    epochs = 5


    for epoch in range(epochs):

        model.train()

        total_loss = 0
        correct = 0
        total = 0


        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()


        acc = 100 * correct / total

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {total_loss:.4f} "
            f"Accuracy: {acc:.2f}%"
        )

    os.makedirs("models", exist_ok=True)  

    model_path = "models/brain_model.pth"

    torch.save(model.state_dict(), model_path)

    print("Model saved successfully at:", model_path)


if __name__ == "__main__":
    main()
