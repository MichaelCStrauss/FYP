# %%
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os
import torchvision.models.detection
from fyp.data.flickr8k import Flickr8kDataModule

# %%
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# %%

os.chdir('/home/michael/Documents/fyp')

datamodule = Flickr8kDataModule()
datamodule.prepare_data()
datamodule.setup()

dataloader = datamodule.train_dataloader()

image, caption = iter(dataloader).next()

# %%
npimg = image[4].numpy()
plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')


# %%
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.cuda()

# %%
image = image.to('cuda')
squeezed = image[0].unsqueeze(0)
print(squeezed.shape)
out = model(squeezed)
# for label, score in zip(out['labels'], out['scores']):
#     print(f'{COCO_INSTANCE_CATEGORY_NAMES[label]}: {score}')
# %%

print(out[0].shape)
for layer in out[1]: 
    print(layer.shape)
# %%

# %%
