from torch.utils.data import Dataset
import os
import torch
import torchnlp.encoders.text


class Flickr8kLMDataset(Dataset):
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