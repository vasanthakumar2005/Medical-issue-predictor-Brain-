import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class BrainDataset(Dataset):

    def __init__(self, root_dir):

        self.image_paths = []
        self.labels = []

        # Image Transform (IMPORTANT: Same for train & predict)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

        # Dataset Paths
        ct_path = os.path.join(root_dir, "Brain Tumor CT scan Images")
        mri_path = os.path.join(root_dir, "Brain Tumor MRI images")

        # Load both folders
        self._load_images(ct_path)
        self._load_images(mri_path)

        print(f"Found images: {len(self.image_paths)}")


    def _load_images(self, base_path):

        healthy_path = os.path.join(base_path, "Healthy")
        tumor_path = os.path.join(base_path, "Tumor")

        # Healthy = 0
        if os.path.exists(healthy_path):

            for img in os.listdir(healthy_path):

                if img.lower().endswith((".jpg", ".jpeg", ".png")):

                    self.image_paths.append(
                        os.path.join(healthy_path, img)
                    )

                    self.labels.append(0)


        # Tumor = 1
        if os.path.exists(tumor_path):

            for img in os.listdir(tumor_path):

                if img.lower().endswith((".jpg", ".jpeg", ".png")):

                    self.image_paths.append(
                        os.path.join(tumor_path, img)
                    )

                    self.labels.append(1)


    def __len__(self):

        return len(self.image_paths)


    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_dataset():

    # DO NOT CHANGE THIS PATH
    data_path = "data/Dataset"

    return BrainDataset(data_path)
