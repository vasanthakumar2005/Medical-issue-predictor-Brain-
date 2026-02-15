import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys
import os

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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

def predict(image_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BrainCNN().to(device)

    model.load_state_dict(
        torch.load("models/brain_model.pth", map_location=device)
    )

    model.eval()

    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  
    image = image.to(device)

    with torch.no_grad():

        outputs = model(image)

        _, predicted = torch.max(outputs, 1)


    classes = ["Healthy", "Tumor"]

    return classes[predicted.item()]
if __name__ == "__main__":

    if len(sys.argv) != 2:

        print("Usage: python predict.py <image_path>")
        sys.exit()


    image_path = sys.argv[1]


    if not os.path.exists(image_path):

        print("Image not found!")
        sys.exit()


    result = predict(image_path)

    print("Prediction:", result)
