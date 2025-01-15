import os
from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
import torchvision
import base64
from io import BytesIO


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


app = Flask(__name__)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    num_classes = checkpoint['model_state_dict']['classification_head.weight'].shape[0]

    model = MetricFeatureExtractor(num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


MODEL_PATH = 'best_metric_model.pth'
FEATURES_PATH = 'features_and_paths.pkl'
IMAGE_FOLDER = 'static/dataset/Mon_An_Ha_Noi_Split/train'

model = load_model(MODEL_PATH)
with open(FEATURES_PATH, 'rb') as f:
    data = pickle.load(f)
    features = data['features']
    image_paths = data['image_paths']


def search_similar_images(query_image, top_k=5):
    img_tensor = transform(query_image).unsqueeze(0)

    with torch.no_grad():
        query_feature = model.extract_features(img_tensor).numpy()

    similarities = cosine_similarity(query_feature, features)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = [(image_paths[i], similarities[i]) for i in top_indices]

    return results


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'})

    try:
        image = Image.open(file).convert('RGB')
        results = search_similar_images(image)

        search_results = []
        for path, score in results:
            with open(path, 'rb') as img_file:
                img = Image.open(img_file)
                img = img.resize((200, 200))
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

            search_results.append({
                'image': img_str,
                'score': float(score),
                'path': path
            })

        return jsonify({'results': search_results})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)