import typer
import requests
import os
from fyp.data.download_file import download_file

data_path = "data/external/flickr8k/data.zip"
label_path = "data/external/flickr8k/labels.zip"

def download():
    if not (
        os.path.exists("data/external/flickr8k/data.zip")
        and os.path.exists("data/external/flickr8k/labels.zip")
    ):
        _download_raw()


def _download_raw():
    if not os.path.exists("data/external/flickr8k"):
        os.mkdir("data/external/flickr8k")
    download_file(
        "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip",
        data_path,
    )
    download_file(
        "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip",
        label_path,
    )


if __name__ == "__main__":
    typer.run(download)