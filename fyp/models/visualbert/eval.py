# %%
import os
from dotenv import load_dotenv
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

os.chdir("/home/michael/Documents/fyp")
from fyp.models.visualbert.model import VisualBERT
from fyp.data.flickr8k import Flickr8kDataModule, DatasetType


load_dotenv()

# %%
data = Flickr8kDataModule(DatasetType.MaskedLanguageModel)
data.prepare_data()
data.setup()

model = VisualBERT.load_from_checkpoint(
    "models/visualbert/wandb/latest-run/files/final-year-project/2h193tcx/checkpoints/epoch=8.ckpt",
)
model.cuda()
model.eval()

# %%
dataloader = data.val_dataloader()

iterable = iter(dataloader)
targets, batch_features, vision_masks = next(iterable)

target, features, vision_mask = (
    targets[0],
    batch_features.split(1)[0],
    vision_masks.split(1)[0],
)

generated = model.inference(features.to('cuda'), vision_mask.to('cuda'), 20)

print(f"{generated=}, {target=}")

# %%
