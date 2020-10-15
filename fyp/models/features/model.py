import torch
import torch.nn as nn
import detectron2.data.transforms as T


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self._create_model()

    def _create_model(self):
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
        cfg.TEST.DETECTIONS_PER_IMAGE = 8
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )

        model = build_model(cfg)
        model.eval()

        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.model = model
        self.cfg = cfg

    def forward(self, images):
        outputs = []
        for image_r in images.split(1):
            image_r = image_r.squeeze().cpu().numpy()
            aug = T.ResizeShortestEdge(
                [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST],
                self.cfg.INPUT.MAX_SIZE_TEST,
            )

            image = aug.get_transform(image_r).apply_image(image_r)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).cuda()
            height, width = image.shape[:2]
            inputs = [{"image": image, "height": height, "width": width}]

            with torch.no_grad():
                images = self.model.preprocess_image(inputs)  # don't forget to preprocess
                features = self.model.backbone(images.tensor)  # set of cnn features
                proposals, _ = self.model.proposal_generator(images, features, None)  # RPN

                features_ = [features[f] for f in self.model.roi_heads.box_in_features]
                box_features = self.model.roi_heads.box_pooler(
                    features_, [x.proposal_boxes for x in proposals]
                )
                box_features = self.model.roi_heads.box_head(
                    box_features
                )  # features of all 1k candidates
                predictions = self.model.roi_heads.box_predictor(box_features)
                pred_instances, pred_inds = self.model.roi_heads.box_predictor.inference(
                    predictions, proposals
                )
                pred_instances = self.model.roi_heads.forward_with_given_boxes(
                    features, pred_instances
                )

                # output boxes, masks, scores, etc
                pred_instances = self.model._postprocess(
                    pred_instances, inputs, images.image_sizes
                )  # scale box to orig size
                # features of the proposed boxes
                feats = box_features[pred_inds]

                # self.print_label(pred_instances)

                outputs.append(feats)

        return outputs

    def print_label(self, pred_instances):
        from detectron2.data.catalog import MetadataCatalog

        catalog = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        labels = catalog.get("thing_classes")

        for idx, img in enumerate(pred_instances):
            instances = img["instances"]
            classes = [labels[i] for i in instances.pred_classes]
            print(f"image contains {', '.join(classes)}")
