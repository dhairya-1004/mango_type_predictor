import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn.functional as F

from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn.functional as F

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Classes
class_names = ['Dasheri', 'Kesar', 'Langdo', 'Rajapuri', 'Totapuri']

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load datasets
data_dir = "mango dataset"
dataset = datasets.ImageFolder(data_dir, transform=train_transform)

# Split into train and val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Update val_dataset transform
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Model
model = models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.4f}")

# Save model
torch.save(model.state_dict(), "mango_model_5class.pth")
# Evaluation
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\n📊 Classification Report:\n")
print(classification_report(all_labels, all_preds, target_names=class_names))

# Class names
known_classes = ['Dasheri', 'Kesar', 'Langdo', 'Rajapuri', 'Totapuri']
class_names = known_classes + ['Others']  # Add 'Others' as final output class

# Load model
model = models.resnet18(weights="DEFAULT")
model.fc = torch.nn.Linear(model.fc.in_features, len(known_classes))  # Only trained on 5
model.load_state_dict(torch.load("mango_model_5class.pth", map_location=torch.device('cpu')))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load the image
image_path = r"testimages\kesar3.jpg"  # Change to your file
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Predict with confidence
with torch.no_grad():
    output = model(input_tensor)
    probabilities = F.softmax(output, dim=1)
    confidence, predicted_class_idx = torch.max(probabilities, 1)
    confidence_value = confidence.item()

# Set confidence threshold (tune this if needed)
threshold = 0.70
if confidence_value < threshold:
    predicted_class = "Others"
else:
    predicted_class = known_classes[predicted_class_idx.item()]

print(f"🟡 Confidence: {confidence_value:.2f}")
print(f"🟢 Predicted Mango Variety: {predicted_class}")

# Class names (trained on these only)
known_classes = ['Dasheri', 'Kesar', 'Langdo', 'Rajapuri', 'Totapuri']
class_names = known_classes + ['Others']

# Load model
model = models.resnet18(weights="DEFAULT")
model.fc = torch.nn.Linear(model.fc.in_features, len(known_classes))
model.load_state_dict(torch.load("mango_model_5class.pth", map_location=torch.device('cpu')))
model.eval()

# Correct normalization (based on ImageNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load test image
image_path = r"C:/Users/Maharshi patel/AI_Hackathon/mango dataset/Dosehri/IMG_20210629_183104.jpg"  # Change this
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    probs = F.softmax(output, dim=1)[0]
    top_prob, top_class = torch.max(probs, 0)

threshold = 0.7
if top_prob.item() < threshold:
    predicted_label = "Others"
else:
    predicted_label = known_classes[top_class.item()]

# 🔍 Debug info
print(f"🔎 Class Probabilities: {[f'{c}: {p:.2f}' for c, p in zip(known_classes, probs.tolist())]}")
print(f"🟢 Predicted Mango Variety: {predicted_label} (Confidence: {top_prob.item():.2f})")
