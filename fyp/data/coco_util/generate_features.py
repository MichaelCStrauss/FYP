def generate_features(
    model,
    transform,
    images="./data/raw/coco/train2017",
    annotations="./data/raw/coco/annotations/captions_train2017.json",
):
    import torch
    import os
    import torchvision.datasets
    from tqdm import tqdm

    dataset = torchvision.datasets.CocoCaptions(
        root=images, annFile=annotations, transform=transform
    )

    files = os.listdir("./data/processed/coco/features/")
    indices = [int(file.split(".")[0]) for file in files]
    indices = sorted(indices)
    if len(indices) > 0:
        last_idx = indices[-1]
    else:
        last_idx = 0
    print(f"Starting from index {last_idx=}")
    dataset = torch.utils.data.Subset(dataset, list(range(last_idx, len(dataset))))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=8)

    for idx, (images, captions) in enumerate(tqdm(iter(dataloader))):
        idx += last_idx
        if os.path.exists(f"./data/processed/coco/features/{idx}.pt"):
            continue
        features_set = model(images.cuda())
        for i, features in enumerate(features_set):
            features = features.cpu()
            torch.save(features, f"./data/processed/coco/features/{idx}.pt")
            image_caps = [c[i] for c in captions]
            torch.save(image_caps, f"./data/processed/coco/captions/{idx}.pt")


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