def generate_features(
    model,
    transform,
    images="./data/processed/coco/train2017",
    annotations="./data/processed/coco/annotations/captions_train2017.json",
):
    import torch
    import torchvision.datasets
    from detectron2.utils.visualizer import Visualizer

    dataset = torchvision.datasets.CocoCaptions(
        root=images, annFile=annotations, transform=transform
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=8)

    for images, captions in iter(dataloader):
        features = model(images.cuda())
        print(features)
        break


if __name__ == "__main__":
    from torchvision import transforms
    from fyp.models.features.model import FeatureExtractor

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
    
    model = FeatureExtractor()
    model = model.cuda()
    model.eval()

    generate_features(model, load_image)