import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import torchvision.transforms as T


class ImageAttributeDataset(Dataset):
    def __init__(self, image_dir, csv_path, image_size=64):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir

        self.feature_cols = [c for c in self.df.columns if c.startswith("has_") or c.startswith("is_")]
        self.num_features = len(self.feature_cols)

        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        fname = str(row["file_name"])
        img_path = os.path.join(self.image_dir, fname)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing image file: {img_path}")

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        attrs = row[self.feature_cols].fillna(0).astype("float32").values
        attrs = torch.from_numpy(attrs)

        return image, attrs


def get_dataset(image_dir, csv_path, batch_size, image_size=64, num_workers=2):
    dataset = ImageAttributeDataset(image_dir=image_dir, csv_path=csv_path, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    return loader, dataset.num_features
