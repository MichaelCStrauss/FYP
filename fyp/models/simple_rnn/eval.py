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
dataloader = data.val_dataloader()

features, captions, lengths, targets = iter(dataloader).next()

def generate_desc(feature):
    caption = "<s>"
    max_length = 20
    for i in range(max_length):
        encoded = data.encoder.encode(caption)
        encoded = encoded.unsqueeze(0)
        output = model(feature.cuda(), encoded.cuda(), None)
        output = output.squeeze(0)
        idx = torch.argmax(output)
        word = data.encoder.decode([int(output[idx])])
        caption += ' ' + word

        if word == "<end>":
            break
    return caption


for feature in features.split(1):
    print(feature[0, :2])
    description = generate_desc(feature)
    print(description)
