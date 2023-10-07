from flask import Flask, render_template, request, jsonify
import os
import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import pandas as pd
import numpy as np
import csv
from test import TableQuestionAnswering


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

tqa_instance = TableQuestionAnswering()
tqa_instance.load_table('Model/DiseaseChatbotData.csv')


@app.route('/', methods=['GET', 'POST'])
def home():
    answer = None  # Initialize answer as None
    if request.method == 'POST':
        query = request.form.get('user_input')
        print(query)  # Check if 'query' is printed correctly
        answer = tqa_instance.answer_query(query)
        print(answer)  # Check if 'answer' is printed correctly
    return render_template('home.html', answer=answer)




@app.route('/index')
def ai_detect_page():
    return render_template('index.html')

@app.route('/supplement')
def supplement():
   # Read data from the CSV file
    supplement_data = []
    with open('Model\supplement_info.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            supplement_data.append({
                'supplement': row['supplement name'],
                'supplement_img': row['supplement image'],
                'supplement_prod_link': row['buy link']
            })

    return render_template('supplement.html', supplement_data=supplement_data)

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
            supplement = suppliment_info['supplement name'][pred]
            supplement_img = suppliment_info['supplement image'][pred]
            supplement_prod_link = suppliment_info['buy link'][pred]

            image_url = '/' + file_path

            # Return the prediction as JSON (customize response as needed)
            response = {
                'prediction': title,
                'image':image_url,
                'discrption':disease_info['description'][pred],
                'possible_step':disease_info['Possible Steps'][pred],
                'supplement':supplement,
                'supplement_img':supplement_img,
                'supplement_name':suppliment_info['supplement name'][pred],
                "supplement_prod_link":supplement_prod_link
                }
            # return jsonify(response)
            return render_template('submit.html',data=response)
    
    # Handle errors or invalid input here (customize as needed)
    return jsonify({'error': 'Invalid request'})

@app.route('/response',methods=['GET', 'POST'])
def response():
    answer = None  # Initialize answer as None
    if request.method == 'POST':
        query = request.form.get('text')
        print(query)  # Check if 'query' is printed correctly
        answer = tqa_instance.answer_query(query)
        print(answer)  # Check if 'answer' is printed correctly
    return render_template('chatbot.html',answer=answer)

@app.route('/learnmore')
def learnmore():
    return render_template('learnmore.html')

if __name__ == '__main__':
    app.run(debug=True)
