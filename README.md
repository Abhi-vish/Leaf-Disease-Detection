
# Leaf Disease Detection

Leaf disease detection is a crucial web app in agriculture, focusing on the automated identification and diagnosis of diseases and stress conditions affecting plant leaves. By analyzing images of leaves for patterns and symptoms of diseases or nutrient deficiencies, this technology-enabled web application enables early detection, precise treatment, and the promotion of sustainable farming practices. It empowers farmers with valuable insights into plant health, contributing to improved crop yields, reduced environmental impact, and enhanced food security.
## Dataset

This dataset contains 87,000 RGB images of healthy and diseased crop leaves categorized into 38 classes. It follows an 80/20 training-validation split, preserving the directory structure, and includes 33 test images for predictions. It's a vital resource for agricultural research and machine learning applications in crop health monitoring and disease detection.

	0 : 'Apple scab',
	1 : 'Apple black rot',
	2 : 'Apple cedar apple rust',
	3 : 'Apple healthy',
	4 : 'Blueberry healthy',
	5 : 'Cherry powdery mildew',
	6 : 'Cherry healthy',
	7 : 'Corn Cercospora leaf Gray leaf spot',
	8 : 'Corn common rust',
	9 : 'Corn Northen leaf blight',
	10 : 'Corn healthy',
	11 : 'Grap black rot',
	12 : 'Grap esca (black measles)',
	13 : 'Grap leaf blight (Isaropsis leaf spot)',
	14 : 'Grap healthy',
	15 : 'Orange Haunglonbing (Citrus greening)',
	16 : 'Peach Bacterial spot',
	17 : 'Peach healthy',
	18 : 'Pepper bell backterial spot',
	19 : 'Pepper bell healthy',
	20 : 'Potato early blight',
	21 : 'Potato late blight',
	22 : 'Potato healthy',
	23 : 'Raspberry healthy',
	24 : 'Soybean healthy',
	25 : 'Squash Powdery mildew',
	26 : 'Strawberry leaf scorch',
	27 : 'Strawberry healthy',
	28 : 'Tomato bacterial spot',
	29 : 'Tomato early blight',
	30 : 'Tomato late blight',
	31 : 'Tomato leaf mold',
	32 : 'Tomato septoria leaf spot',
	33 : 'Tomato spider mites two spotted spider mite',
	34 : 'Tomato target spot',
	35 : 'Tomato yellow leaf curl virus',
	36 : 'Tomato mosaic virus',
	37 : 'Tomato healthy'

## preview



#### 1. Disease Detection 

![Untitled video - Made with Clipchamp (1)](https://github.com/Abhi-vish/Leaf-Disease-Detection/assets/109618783/351e1a1d-aa5c-41dd-9945-2704a0d17e82)

#### 2. ChatBot

![Untitled video - Made with Clipchamp (2)](https://github.com/Abhi-vish/Leaf-Disease-Detection/assets/109618783/4f59d686-c8f5-4581-bb41-d4967902a0d3)


## Virtual Environment 



Create a virtual environment to ensure that the project runs smoothly without any impact on your system's environment.
```bash
python -m venv myenv
```
And activate virtual environment by running command
```bash
myenv\Scripts\activate
```

## Installation

#### Install my-project in your envrionment

step1 : clone the repo
```bash
https://github.com/Abhi-vish/Leaf-Disease-Detection.git
```

step2 : install requirements.txt package by running commnad
```bash
pip install  -r requirements.txt
```
step3 : open terminal and run commnad
```bash
python app.py 
```    
    
## 🚀 About Me
I'm a student...


## Lessons Learned

While building this project, I learned several key concepts and faced various challenges. Here are the details:

### Data Handling:

* Loading and Preprocessing Training and Validation Data.
* Converting Images into Tensors for Machine Learning.
### Device Agnostic Code:

* Writing Code Compatible with Different Hardware (Device Agnostic).
### Transfer Learning:

* Utilizing Pretrained Models for Improved Model Performance.
* Fine-Tuning Pretrained Models on Custom Datasets.
### Model Evaluation:

* Techniques for Evaluating Model Performance.
### Model Management:

* Saving and Managing Trained Models.
### GUI Development:

* Building Graphical User Interfaces (GUIs) for Model Interaction.
* Integrating Machine Learning Models with GUIs.
