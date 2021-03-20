# %%
from fyp.models.visualbert.config import TrainingObjective
import typer
import os
from dotenv import load_dotenv
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from torch.utils.data import Subset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as tv
from fyp.models.visualbert.model import VisualBERT
from fyp.data.coco_captions import CocoCaptions, CocoCaptionsDataset
from fyp.models.features.model import FeatureExtractor
import PIL
import math
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize


load_dotenv()


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


# %%
def evaluate(image: str = None):
    model = VisualBERT.load_from_checkpoint(
        "models/electra/final-year-project/3osm0cr3/checkpoints/epoch=17-step=231535.ckpt",
        manual_lm_head=True
    )
    model.training_objective = TrainingObjective.Captioning
    model.cuda()
    model.eval()

    feature_model = FeatureExtractor()
    feature_model.cuda()
    feature_model.eval()

    if image is None:
        data = CocoCaptions()
        data.prepare_data()
        data.setup()
        images = "./data/raw/coco/train2017"
        annotations = "./data/raw/coco/annotations/captions_train2017.json"
        raw_dataset = tv.CocoCaptions(
            root=images, annFile=annotations, transform=load_image
        )
        raw_subset = Subset(raw_dataset, data.val.indices)
        dataloader = DataLoader(raw_subset, batch_size=16, num_workers=0)

        # %%
        batch = next(iter(dataloader))
        images, caption_sets = batch
    else:
        image = PIL.Image.open(image)
        images = load_image(image).unsqueeze(0).cuda()
        caption_sets = []

    fig = plt.figure()
    num_rows = math.ceil(math.sqrt(images.shape[0]))
    num_cols = num_rows
    for i, image in enumerate(images.split(1)):
        features = feature_model(image)[0]
        features, mask = CocoCaptionsDataset.postprocess_features(features, 8)
        features = features.unsqueeze(0)
        mask = mask.unsqueeze(0)
        generated = model.inference(features.cuda(), mask.cuda(), 20)
        targets = [x[i] for x in caption_sets]
        ax = fig.add_subplot(num_rows, num_cols, i + 1)
        plt.imshow(image.cpu().squeeze().flip(2) / 255)

        ax.set_xlabel(generated)

        bleu = sentence_bleu(
            [word_tokenize(x) for x in targets], word_tokenize(generated)
        )

        print(f"{generated=}, {targets=}, {bleu}")
    plt.show()


if __name__ == "__main__":
    typer.run(evaluate)
