# %%
# Some basic setup:
# Setup detectron2 logger

# import some common libraries


def create_model():
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import build_model

    # Create a config
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    cfg.TEST.DETECTIONS_PER_IMAGE = 5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )

    model = build_model(cfg)
    model.eval()

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    return model, cfg


def get_features(model, file, cfg):
    import cv2
    import torch
    import detectron2.data.transforms as T
    from detectron2.data.catalog import MetadataCatalog

    catalog = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    labels = catalog.get('thing_classes')

    image_r = cv2.imread(file)

    if image_r is None:
        raise ValueError("File not found: " + file)

    aug = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )

    image = aug.get_transform(image_r).apply_image(image_r)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    height, width = image_r.shape[:2]

    inputs = [{"image": image, "height": height, "width": width}]

    with torch.no_grad():
        images = model.preprocess_image(inputs)  # don't forget to preprocess
        features = model.backbone(images.tensor)  # set of cnn features
        proposals, _ = model.proposal_generator(images, features, None)  # RPN

        features_ = [features[f] for f in model.roi_heads.box_in_features]
        box_features = model.roi_heads.box_pooler(
            features_, [x.proposal_boxes for x in proposals]
        )
        box_features = model.roi_heads.box_head(
            box_features
        )  # features of all 1k candidates
        predictions = model.roi_heads.box_predictor(box_features)
        pred_instances, pred_inds = model.roi_heads.box_predictor.inference(
            predictions, proposals
        )
        pred_instances = model.roi_heads.forward_with_given_boxes(
            features, pred_instances
        )

        # output boxes, masks, scores, etc
        pred_instances = model._postprocess(
            pred_instances, inputs, images.image_sizes
        )  # scale box to orig size
        # features of the proposed boxes
        feats = box_features[pred_inds]
        instances = pred_instances[0]['instances']
        classes = [labels[i] for i in instances.pred_classes]
        print(f"{file}: contains {', '.join(classes)}. {feats.shape=}")
        return feats
