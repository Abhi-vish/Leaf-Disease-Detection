"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_setup, engine, model_builder, utils
from pathlib import Path
from torchvision import transforms
import torchvision.models as models

# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# Define data directories
data_dir = "/content/plant_disease_dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
train_dir = Path(data_dir) / 'train'
valid_dir = Path(data_dir) / 'valid'

# Define data transformations
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = 'cuda' if torch.cuda.is_available else 'cpu'

# Create train/ test dataloader and get class names as a list
train_dataloader, valid_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    valid_dir=valid_dir,
    train_transform=train_transform,
    valid_transform=valid_transform,
    batch_size=32,
    num_workers=os.cpu_count()
)


# Create model with help from model_builder.py
model = models.resnet18(pretrained=True)
model = model_builder.loadModel(
    model,
    num_classes=len(class_names)
).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             valid_dataloader=valid_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name="LDD.pth")
