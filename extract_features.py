import os
import torch
from torchvision import transforms
from PIL import Image
import pickle
import torch.nn as nn
import torchvision


class MetricFeatureExtractor(nn.Module):
    def __init__(self, num_classes, embedding_dim=2048):
        super().__init__()
        resnet = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.projection = nn.Sequential(
            nn.Linear(2048, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU()
        )

        self.classification_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        features = self.global_pool(features)
        features = features.view(features.size(0), -1)
        embeddings = self.projection(features)
        return embeddings

    def extract_features(self, images):
        return self.forward(images)


def extract_features_from_images(model, dataset_dir, transform):
    features = []
    image_paths = []

    for class_dir in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_dir)
        if os.path.isdir(class_path):
            print(f"Processing {class_dir}...")
            for img_name in os.listdir(class_path):
                if img_name.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(class_path, img_name)
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img_tensor = transform(img).unsqueeze(0)
                        with torch.no_grad():
                            feature = model.extract_features(img_tensor).numpy()
                            features.append(feature[0])
                            image_paths.append(img_path)
                    except Exception as e:
                        print(f"Error processing {img_path}: {str(e)}")

    return features, image_paths


def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    MODEL_PATH = 'best_metric_model.pth'
    DATASET_DIR = 'static/dataset/Mon_An_Ha_Noi_Split/train'

    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    num_classes = checkpoint['model_state_dict']['classification_head.weight'].shape[0]

    model = MetricFeatureExtractor(num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    features, image_paths = extract_features_from_images(model, DATASET_DIR, transform)

    with open("features_and_paths.pkl", "wb") as f:
        pickle.dump({
            "features": features,
            "image_paths": image_paths
        }, f)

    print(f"Done! Processed {len(image_paths)} images")
    print(f"Features shape: {len(features)}x{len(features[0])}")


if __name__ == "__main__":
    main()