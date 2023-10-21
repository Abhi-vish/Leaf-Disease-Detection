import torch
import torch.nn as nn
import torchvision.models as models

def loadModel(model, num_classes):
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
