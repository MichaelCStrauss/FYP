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
    "models/visualbert/wandb/latest-run/files/final-year-project/11un49jp/checkpoints/epoch=9.ckpt",
)
model.cuda()
model.eval()

# %%
dataloader = data.val_dataloader()

iterable = iter(dataloader)


batch = next(iterable)
targets, batch_features, vision_masks, filenames = batch

fig = plt.figure()
for i, (target, features, vision_mask, file) in enumerate(
    zip(targets, batch_features.split(1), vision_masks.split(1), filenames)
):
    generated = model.inference(features.to("cuda"), vision_mask.to("cuda"), 20)
    ax = fig.add_subplot(len(filenames) // 3 + 1, 3, i + 1)
    img = mpimg.imread("data/processed/flickr8k/images/" + file)
    plt.imshow(img)

    ax.set_xlabel(generated)

    print(f"{generated=}, {target=}")
plt.show()

# %%
