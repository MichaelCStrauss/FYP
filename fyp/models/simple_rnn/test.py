# %%
import os
from dotenv import load_dotenv
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import CometLogger


from fyp.models.simple_rnn.model import SimpleRNNCaptioner
from fyp.data.flickr8k import Flickr8kDataModule

os.chdir("/home/michael/Documents/fyp")

load_dotenv()

# %%
data = Flickr8kDataModule()
data.prepare_data()
data.setup()

model = SimpleRNNCaptioner.load_from_checkpoint(
    "models/simple_rnn/final-year-project/1855a56dcb4a425cb11bb33346b94b9c/checkpoints/epoch=6.ckpt",
)
model.cuda()
model.eval()

# %%
dataloader = data.train_dataloader()

features, captions, lengths, targets = iter(dataloader).next()


y_hat = model(features.cuda(), captions.cuda(), lengths.cuda())

for caption, target, y_hat, length in zip(
    captions.split(1), targets.split(1), y_hat.split(1), lengths.split(1)
):
    out = y_hat.squeeze(0).argmax()
    caption = caption.squeeze(0)[-int(length):]
    print(f"{caption=} {target=} {out=}")
