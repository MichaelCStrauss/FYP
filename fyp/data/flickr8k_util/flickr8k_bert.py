from torch.utils.data import Dataset
import os
import torch
from transformers import BertTokenizerFast
import torch.nn.functional as F


class Flickr8kBertDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        labels_file: str,
        features_dir: str,
        masked_language_model: bool = False,
        padded_image_length: int = 5,
    ):
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
            self.examples.append((filename, caption))

        self.feature_cache = {}
        self.padded_image_length = padded_image_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        file, caption = self.examples[i]

        features = None
        if file in self.feature_cache:
            features = self.feature_cache[file]
        else:
            features = torch.load(os.path.join(self.features_dir, file + ".pt"))
            self.feature_cache[file] = features

        num_features = features.shape[0]
        pad_amount = self.padded_image_length - num_features
        features = F.pad(features, (0, 0, 0, pad_amount))

        mask = torch.ones((self.padded_image_length,))
        mask[num_features:] = 0

        return caption, features, mask
