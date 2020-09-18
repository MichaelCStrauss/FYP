from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import os
from fyp.data.flickr8k_util.download import download
from fyp.data.flickr8k_util.process import unzip, save_features
import torch
from PIL import Image
import torchvision.transforms as transforms
from collections import Counter
import torchnlp.encoders.text
import lru


class Flickr8kDataset(Dataset):
    def __init__(self, images_dir: str, labels_file: str, features_dir: str):
        self.images_dir = images_dir
        self.features_dir = features_dir
        self.image_files = os.listdir(images_dir)
        self.labels_file = open(labels_file)

        self.length = 0
        self.labels = []
        self.files = []
        self.examples = []

        for line in self.labels_file:
            tsv = line.split("\t")

            filename, number = tsv[0].split("#")

            if not os.path.exists(os.path.join(self.images_dir, filename)):
                continue

            caption: list = tsv[1].strip()
            self.files.append(filename)
            self.labels.append(caption)

        self.encoder = torchnlp.encoders.text.WhitespaceEncoder(
            self.labels, append_eos=True, min_occurrences=3
        )

        def spaces(x: str):
            return x.count(" ")

        max_str = max(self.labels, key=spaces)
        self.max_length = spaces(max_str) + 2

        for label, file in zip(self.labels, self.files):
            tokenized = list(self.encoder.encode("<s> " + label))
            for i in range(1, len(tokenized)):
                caption = tokenized[:i]
                while len(caption) < self.max_length:
                    caption.insert(0, 0)
                target = tokenized[i]

                self.examples.append((file, caption, i, target))

        self.feature_cache = {}

        # for file in self.files:
        #     features = torch.load(os.path.join(self.features_dir, file + ".pt"))
        #     self.feature_cache[file] = features


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

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        file, caption, length, target = self.examples[i]

        features = None
        if file in self.feature_cache:
            features = self.feature_cache[file]
        else:
            features = torch.load(os.path.join(self.features_dir, file + ".pt"))
            self.feature_cache[file] = features
        tokens = torch.tensor(caption)
        return features, tokens, length, target, file


class Flickr8kDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.external_dir = "data/external/flickr8k"
        self.interim_dir = "data/interim/flickr8k"
        self.processed_dir = "data/processed/flickr8k"

    def get_feature_model(self):
        resnet = torch.hub.load(
            "pytorch/vision:v0.6.0", "resnet50", pretrained=True
        )
        resnet.eval()
        features = torch.nn.Sequential(*(list(resnet.children())[:-1]))
        features.eval()

        return features, resnet

    def prepare_data(self):
        if not os.path.exists(os.path.join(self.processed_dir, "labels.txt")):
            download()
            unzip()

        if not os.path.exists(os.path.join(self.processed_dir, "features")):
            features, resnet = self.get_feature_model()

            save_features(features, None, Flickr8kDataset.load_image)

    def setup(self, stage=None):

        full = Flickr8kDataset(
            os.path.join(self.processed_dir, "images"),
            os.path.join(self.processed_dir, "labels.txt"),
            os.path.join(self.processed_dir, "features"),
        )
        self.train, self.val, self.test = random_split(
            full,
            [len(full) - 1000, 500, 500],
            generator=torch.Generator().manual_seed(42),
        )
        self.encoder = full.encoder

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=64, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=8)


if __name__ == "__main__":
    datamodule = Flickr8kDataModule()
    datamodule.prepare_data()
    datamodule.setup()