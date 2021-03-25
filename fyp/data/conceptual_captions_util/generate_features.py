def generate_features(
    model,
    transform,
    images="./data/raw/conceptual-captions/",
    annotations_file="./data/raw/conceptual-captions/results.csv",
):
    import torch
    import os
    import torchvision.datasets
    from tqdm import tqdm
    import pandas as pd

    files = os.listdir("./data/processed/conceptual-captions/features/")
    indices = [int(file.split(".")[0]) for file in files]
    indices = sorted(indices)
    if len(indices) > 0:
        last_idx = indices[-1]
    else:
        last_idx = 0
    print(f"Starting from index {last_idx=}")

    annotations: pd.DataFrame = pd.read_csv(annotations_file)

    for row in tqdm(annotations.loc[last_idx:].itertuples()):
        image = transform(os.path.join(images, row.file))
        idx = row.Index
        if os.path.exists(f"./data/processed/conceptual-captions/features/{idx}.pt"):
            continue
        features_set = model(image.cuda())
        for i, features in enumerate(features_set):
            features = features.cpu()
            torch.save(features, f"./data/processed/conceptual-captions/features/{idx}.pt")
            image_caps = [row.caption]
            torch.save(image_caps, f"./data/processed/conceptual-captions/captions/{idx}.pt")


if __name__ == "__main__":
    from PIL import Image
    from torchvision import transforms
    from fyp.models.features.model import FeatureExtractor

    def load_image(image_file):
        image = Image.open(image_file)
        preprocess = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ]
        )
        image_tensor = preprocess(image)

        image_tensor = image_tensor.permute((1, 2, 0)).flip(2).unsqueeze(0) * 255

        return image_tensor

    model = FeatureExtractor()
    model = model.cuda()
    model.eval()

    generate_features(model, load_image)