from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import os
from fyp.data.flickr8k_util.download import download
from fyp.data.flickr8k_util.process import (
    unzip,
    save_features_resnet,
    save_features_detectron,
)
from fyp.data.flickr8k_util.flickr8k_LM import Flickr8kLMDataset
from fyp.data.flickr8k_util.flickr8k_bert import Flickr8kBertDataset
from fyp.data.generate_features import get_features, create_model
import torch
from PIL import Image
import torchvision.transforms as transforms
from enum import Enum


class DatasetType(Enum):
    LanguageModel = 1
    MaskedLanguageModel = 2


class Flickr8kDataModule(pl.LightningDataModule):
    def __init__(self, dtype: DatasetType = DatasetType.LanguageModel):
        super().__init__()
        self.external_dir = "data/external/flickr8k"
        self.interim_dir = "data/interim/flickr8k"
        self.processed_dir = "data/processed/flickr8k"
        self.type = dtype

    @staticmethod
    def load_image(filename, directory):
        img = Image.open(os.path.join(directory, filename)).convert("RGB")
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        image_tensor = preprocess(img)

        return image_tensor

    def get_feature_model(self):
        if self.type == DatasetType.LanguageModel:
            resnet = torch.hub.load(
                "pytorch/vision:v0.6.0", "resnet50", pretrained=True
            )
            resnet.eval()
            features = torch.nn.Sequential(*(list(resnet.children())[:-1]))
            features.eval()
            return features, resnet
        else:
            self.model, self.cfg = create_model()

            self.run_model = lambda file: get_features(self.model, file, self.cfg)

            return self.model, self.cfg

    def prepare_data(self):
        if not os.path.exists(os.path.join(self.processed_dir, "labels.txt")):
            download()
            unzip()

        if not os.path.exists(os.path.join(self.processed_dir, "features")):
            _, _ = self.get_feature_model()

            save_features_detectron(self.run_model, self.cfg)

    def setup(self, stage=None):

        if self.type == DatasetType.LanguageModel:
            full = Flickr8kLMDataset(
                os.path.join(self.processed_dir, "images"),
                os.path.join(self.processed_dir, "labels.txt"),
                os.path.join(self.processed_dir, "features"),
            )
            self.encoder = full.encoder
        elif self.type == DatasetType.MaskedLanguageModel:
            full = Flickr8kBertDataset(
                os.path.join(self.processed_dir, "images"),
                os.path.join(self.processed_dir, "labels.txt"),
                os.path.join(self.processed_dir, "features"),
            )
        self.train, self.val, self.test = random_split(
            full,
            [len(full) - 1000, 500, 500],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=8, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=8, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=8)


if __name__ == "__main__":
    datamodule = Flickr8kDataModule(DatasetType.MaskedLanguageModel)
    datamodule.prepare_data()
    datamodule.setup()
