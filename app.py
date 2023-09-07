from flask import Flask, render_template, request, jsonify
import os
import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import pandas as pd
import numpy as np


# Reading the dataset
disease_info = pd.read_csv('Model/disease_info.csv')
suppliment_info = pd.read_csv("Model/supplement_info.csv", encoding='cp1252')

# device agnostic code
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Create instance of the pretrained model
model = models.resnet18(pretrained=True)

# Modifying the layers
num_classes = 38
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)


# Load the model checkpoint
model_checkpoint_path = 'Model/model.pth'
model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
model.eval()

# Define the transformation for input images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# function for image prediction
def prediction(image_path):
    # Load and preprocess the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    output = model(image)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def ai_detect_page():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        if image:
            filename = image.filename
            file_path = os.path.join('static/uploads', filename)
            image.save(file_path)

            # Perform prediction
            pred = prediction(file_path)
            title = disease_info['disease_name'][pred]

            image_url = '/' + file_path

            # Return the prediction as JSON (customize response as needed)
            response = {
                'prediction': title,
                'image':image_url,
                }
            # return jsonify(response)
            return render_template('submit.html',data=response)
    
    # Handle errors or invalid input here (customize as needed)
    return jsonify({'error': 'Invalid request'})

if __name__ == '__main__':
    app.run(debug=True)
