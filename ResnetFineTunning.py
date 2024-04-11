import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.optim import Adam
import time

# Define data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load dataset (example: CIFAR-10)
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms['train'])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transforms['val'])
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

# Load pre-trained ResNet model
model = torchvision.models.resnet18(pretrained=True)

# Freeze all layers except the final fully connected layer
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully connected layer for fine-tuning
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # Assuming 10 classes for CIFAR-10

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
num_epochs = 5
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")

end_time = time.time()
print(f"Training complete in {(end_time - start_time) // 60:.0f}m {(end_time - start_time) % 60:.0f}s")

# Evaluation on validation set
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Validation Accuracy: {100 * accuracy:.2f}%")
