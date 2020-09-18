# %%
import os
from dotenv import load_dotenv
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from fyp.models.simple_rnn.model import SimpleRNNCaptioner
from fyp.data.flickr8k import Flickr8kDataModule

os.chdir("/home/michael/Documents/fyp")

load_dotenv()

# %%
data = Flickr8kDataModule()
data.prepare_data()
data.setup()

model = SimpleRNNCaptioner.load_from_checkpoint(
    "wandb/run-20200918_145707-103u8og0/files/final-year-project/103u8og0/checkpoints/epoch=2.ckpt",
)
model.cuda()
model.eval()

# %%
dataloader = data.test_dataloader()

iterable = iter(dataloader)
iterable.next()
iterable.next()
iterable.next()
features, captions, lengths, targets, files = iterable.next()


def pad_sequence(tokens):
    token_list = list(tokens)
    while len(token_list) < 20:
        token_list.insert(0, 0)
    return torch.as_tensor(token_list)


def generate_desc(feature):
    caption = [3]
    max_length = 20
    for i in range(max_length):
        encoded = pad_sequence(caption)
        encoded = encoded.unsqueeze(0)
        output = model(feature.cuda(), encoded.cuda(), None)
        output = output.squeeze(0)
        idx = torch.argmax(output)
        caption.append(idx)

        if idx == 2:
            break
    return data.encoder.decode(caption)


fig = plt.figure()
for i, (feature, file) in enumerate(zip(features.split(1), files)):
    description = generate_desc(feature)

    ax = fig.add_subplot(len(files) // 3 + 1, 3, i+1)
    img = mpimg.imread("data/processed/flickr8k/images/" + file)
    plt.imshow(img)

    ax.set_xlabel(description)

plt.show()