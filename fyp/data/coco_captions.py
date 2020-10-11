from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from enum import Enum

class CocoCaptions(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.external_dir = "data/external/coco"
        self.interim_dir = "data/interim/coco"
        self.processed_dir = "data/processed/coco"

    @staticmethod
    def load_image(image):
        preprocess = transforms.Compose(
            [
                transforms.Resize(512),
                transforms.ToTensor(),
            ]
        )
        image_tensor = preprocess(image)

        rotated = image_tensor[:, :, ::-1]

        return rotated

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

        # Use a test-train split with different images in the val set
        final_train_index = len(full) - 1000
        train_indices = list(range(0, final_train_index))
        val_indices = list(range(final_train_index, final_train_index + 1000))

        self.train = torch.utils.data.Subset(full, train_indices)
        self.val = torch.utils.data.Subset(full, val_indices)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=8, num_workers=8, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=8, num_workers=8, shuffle=True)


if __name__ == "__main__":
    datamodule = Flickr8kDataModule(DatasetType.MaskedLanguageModel)
    datamodule.prepare_data()
    datamodule.setup()
