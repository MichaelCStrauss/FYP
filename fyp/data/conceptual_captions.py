from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl
import os
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import pandas as pd


class ConceptualCaptionsDataset(Dataset):
    def __init__(self, spec_file: str, features_directory: str):
        self.feature_length = 8
        self.captions_length = 1
        self.features_directory = features_directory
        self.spec_file = spec_file

        self.file_list = os.listdir(features_directory)

        self.spec = pd.read_csv(spec_file, sep="\t")

    @staticmethod
    def postprocess_features(features, feature_length):
        num_features = features.shape[0]
        pad_amount = feature_length - num_features
        features = F.pad(features, (0, 0, 0, pad_amount))

        mask = torch.ones((feature_length,))
        mask[num_features:] = 0

        return features, mask

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        f_path = os.path.join(self.features_directory, self.file_list[index])
        features = torch.load(f_path)
        captions = torch.load(
            os.path.join(self.captions_directory, self.file_list[index])
        )
        while len(captions) < self.captions_length:
            captions.append(None)
        captions = captions[: self.captions_length]

        features, mask = self.postprocess_features(features, self.feature_length)

        return features, mask, captions


class ConceptualCaptions(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.spec_file = "data/raw/conceptual-captions/results.csv"
        self.external_dir = "data/external/coco"
        self.interim_dir = "data/interim/coco"
        self.processed_dir = "data/processed/conceptual-captions"

    @staticmethod
    def load_image(image):
        preprocess = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ]
        )
        image_tensor = preprocess(image)

        image_tensor = image_tensor.permute((1, 2, 0)).flip(2) * 255

        return image_tensor

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.dataset = ConceptualCaptionsDataset(
            features_directory=self.processed_dir + "/features",
            spec_file=self.spec_file,
        )

        train_length = int(len(self.dataset) * 0.92)
        val_length = len(self.dataset) - train_length
        self.train, self.val = random_split(
            self.dataset,
            [train_length, val_length],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=8, num_workers=8, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=8, num_workers=8, shuffle=False)


if __name__ == "__main__":
    data = ConceptualCaptions()
    data.setup()

    loader = data.train_dataloader()

    print("iter")
    it = iter(loader)
    print("iter done")
    features, mask, captions = next(it)

    print(features.shape)
    print(mask.shape)
    features, mask, captions = next(it)

    print(features.shape)
    print(mask.shape)
