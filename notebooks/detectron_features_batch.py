# %%
# Some basic setup:
# Setup detectron2 logger
import detectron2

# import some common libraries
import numpy as np
import os, json, cv2, random

os.chdir("/home/michael/Documents/fyp")

# import some common detectron2 utilities
import matplotlib.pyplot as plt
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
import torch
from PIL import Image
from torchvision import transforms
import detectron2.data.transforms as T

# %%
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9  # set threshold for this model
cfg.TEST.DETECTIONS_PER_IMAGE = 5
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)

model = build_model(cfg)
model.eval()

checkpointer = DetectionCheckpointer(model)
checkpointer.load(cfg.MODEL.WEIGHTS)

# %%
image_pil = Image.open("data/processed/coco/train2017/000000000192.jpg")

preprocess = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ]
)
image_tensor = preprocess(image_pil) * 255

image_tensor = image_tensor.permute((1, 2, 0)).flip(2)

aug = T.ResizeShortestEdge(
    [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
)

image_r = image_tensor.numpy()
image = aug.get_transform(image_r).apply_image(image_r)
image = torch.as_tensor(image_r.astype("float32").transpose(2, 0, 1))
height, width = image_r.shape[:2]

inputs = [{"image": image, "height": height, "width": width}]
# plt.imshow(image.permute(1, 2, 0).numpy()[:, :, ::-1] / 255.0)

# %%
image_pil2 = Image.open("data/processed/coco/train2017/000000209046.jpg")
image_pil2 = Image.open("data/processed/coco/train2017/000000000192.jpg")
image_tensor2 = preprocess(image_pil2) * 255

image_tensor2 = image_tensor2.permute((1, 2, 0)).flip(2)
image_r = image_tensor2.numpy()
image2 = aug.get_transform(image_r).apply_image(image_r)
image2 = torch.as_tensor(image_r.astype("float32").transpose(2, 0, 1))
height, width = image_r.shape[:2]

inputs.append({"image": image2, "height": height, "width": width})

# %%
with torch.no_grad():
    images = model.preprocess_image(inputs)  # don't forget to preprocess
    features = model.backbone(images.tensor)  # set of cnn features
    proposals, _ = model.proposal_generator(images, features, None)  # RPN

    features_ = [features[f] for f in model.roi_heads.box_in_features]
    box_features_pool = model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
    box_features = model.roi_heads.box_head(box_features_pool)  # features of all 1k candidates
    predictions = model.roi_heads.box_predictor(box_features)
    pred_instances, pred_inds = model.roi_heads.box_predictor.inference(predictions, proposals)
    pred_instances = model.roi_heads.forward_with_given_boxes(features, pred_instances)

    # output boxes, masks, scores, etc
    pred_instances = model._postprocess(pred_instances, inputs, images.image_sizes)  # scale box to orig size
    # features of the proposed boxes
    # feats = box_features[pred_inds]
    # print(feats.shape)
    # print(feats.shape)
    v = Visualizer(
        image_r[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2
    )
    out = v.draw_instance_predictions(pred_instances[1]["instances"].to("cpu"))
    plt.imshow(cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))

    for i, img_inds in enumerate(pred_inds):
        print(img_inds)
        inds = img_inds + 1000 * i
        print(inds)
        feats = box_features[inds]
        print(feats)
# %%
print(pred_instances[1])

# %%
