import os
import torch
from tqdm import tqdm
from loguru import logger
from fyp.data.flickr8k_util.download import label_path, data_path
import zipfile
import shutil


def unzip():
    logger.info("Extracting flickr8k files")
    data = zipfile.ZipFile(label_path)
    data.extractall("data/interim/flickr8klabels")

    data = zipfile.ZipFile(data_path)
    data.extractall("data/interim/flickr8kdata")

    shutil.move(
        "data/interim/flickr8kdata/Flicker8k_Dataset", "data/processed/flickr8k/images"
    )

    shutil.move(
        "data/interim/flickr8klabels/Flickr8k.token.txt",
        "data/processed/flickr8k/labels.txt",
    )


def save_features(model, to_tensor):
    out_path = "data/processed/flickr8k/features/"
    os.mkdir(out_path)
    path = "data/processed/flickr8k/images"

    model.cuda()
    for file in tqdm(os.listdir(path)):
        image = to_tensor(file, path)
        image = image.unsqueeze(0).cuda()
        with torch.no_grad():
            features = model(image)
        features = features.reshape((-1,))
        torch.save(features.cpu(), os.path.join(out_path, file + ".pt"))


if __name__ == "__main__":
    save_features()
