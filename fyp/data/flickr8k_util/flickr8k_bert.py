from torch.utils.data import Dataset
import os
import torch
from transformers import BertTokenizerFast


class Flickr8kBertDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        labels_file: str,
        features_dir: str,
        masked_language_model: bool = False,
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

        for file, caption in zip(self.files, self.labels):
            # caption = "[CLS] " + caption + " [SEP]"
            # tokens = self.tokenizer(caption)
            self.examples.append((file, caption))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        file, caption = self.examples[i]

        # features = None
        # if file in self.feature_cache:
        #     features = self.feature_cache[file]
        # else:
        #     features = torch.load(os.path.join(self.features_dir, file + ".pt"))
        #     self.feature_cache[file] = features
        # tokens = torch.tensor(caption)
        return caption
