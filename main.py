# Importing  Libraries
import os
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import StepLR
import sys

# Metrics and visualization
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc, f1_score
import seaborn as sns
import numpy as np

# Check GPU
print()
print(f"Torch Version: {torch.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize(384, interpolation=InterpolationMode.BICUBIC),  # Resize to 384x384
    transforms.CenterCrop(384), 
    transforms.ToTensor()  # Tensor
    ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalization
])

train_dir = "/home/wooju.chung/organized_TMA2/train"
val_dir = "/home/wooju.chung/organized_TMA2/val"

train_dataset_bfsplit = ImageFolder(root=train_dir, transform=transform)

val_dataset = ImageFolder(root=val_dir, transform=transform)

# stratified split train dataset into train and test set
train_indices, test_indices = train_test_split(
    np.arange(len(train_dataset_bfsplit)),
    test_size=0.2,  # 20%
    stratify=train_dataset_bfsplit.targets,
    random_state=42
)
test_dataset = torch.utils.data.Subset(train_dataset_bfsplit, test_indices)
train_dataset = torch.utils.data.Subset(train_dataset_bfsplit, train_indices)

# create DataLoader
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

# display classes on dataset
def print_dataset_stats(dataset, dataset_name):
    if isinstance(dataset, ImageFolder):
        labels = np.array(dataset.targets)
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Dataset: {dataset_name}")
        for label, count in zip(unique, counts):
            print(f"  {dataset.classes[label]}: {count}")
    elif isinstance(dataset, torch.utils.data.Subset):
        labels = np.array([dataset.dataset.targets[i] for i in dataset.indices])
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Dataset: {dataset_name}")
        for label, count in zip(unique, counts):
            print(f"  {dataset.dataset.classes[label]}: {count}")

# print train, val, test dataset's classes
print_dataset_stats(train_dataset_bfsplit, "Original Train Dataset (Before Split)")
print_dataset_stats(train_dataset, "Train Dataset (After Split)")
print_dataset_stats(test_dataset, "Test Dataset")
print_dataset_stats(val_dataset, "Validation Dataset")

# define efficient net model
model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Sequential(
    nn.Dropout(p=0.5),  # Add dropout for regularization
    nn.Linear(model.classifier[1].in_features, 2)  # Output layer for binary classification
)

model = model.to(device)
print(f"Model Name: {model.__class__.__name__}")

# set loss function, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=3, gamma=0.1)  # reduce learning rate per 3 epoch

# loop setting
num_epochs = 30
best_val_loss = float('inf')
save_path = f"/home/wooju.chung/TMA/output_tma/best_model.pth"

# Early Stopping
patience = 8
no_improvement_epochs = 0  

# Metric tracking lists
train_accuracies, train_precisions, train_recalls, train_f1s, train_losses = [], [], [], [], []
val_accuracies, val_precisions, val_recalls, val_f1s, val_losses = [], [], [], [], []

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print("-" * 20)

    # Training
    model.train()
    train_loss = 0.0
    all_train_labels = []
    all_train_preds = []

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1) 
        all_train_labels.extend(labels.cpu().numpy())
        all_train_preds.extend(preds.cpu().numpy())
    
    train_loss = train_loss / len(train_dataset)
    train_losses.append(train_loss)  # Train Loss

    # Trainset evaluation
    model.eval()
    with torch.no_grad():
        all_train_labels, all_train_preds, all_train_probs = [], [], []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy() 
            preds = torch.argmax(outputs, dim=1).cpu().numpy() 
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(preds)
            all_train_probs.extend(probs)

    # Train metrics
    train_accuracy = accuracy_score(all_train_labels, all_train_preds)
    train_precision = precision_score(all_train_labels, all_train_preds, pos_label=1)
    train_recall = recall_score(all_train_labels, all_train_preds, pos_label=1)
    train_f1 = f1_score(all_train_labels, all_train_preds, pos_label=1)

    train_accuracies.append(train_accuracy)
    train_precisions.append(train_precision)
    train_recalls.append(train_recall)
    train_f1s.append(train_f1)

    print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    all_val_labels = []
    all_val_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_val_labels.extend(labels.cpu().numpy())
            all_val_preds.extend(preds.cpu().numpy())
    
    val_loss = val_loss / len(val_dataset)
    val_losses.append(val_loss)  # Val loss

    # Validation metrics
    val_accuracy = accuracy_score(all_val_labels, all_val_preds)
    val_precision = precision_score(all_val_labels, all_val_preds, pos_label=1)
    val_recall = recall_score(all_val_labels, all_val_preds, pos_label=1)
    val_f1 = f1_score(all_val_labels, all_val_preds, pos_label=1)

    val_accuracies.append(val_accuracy)
    val_precisions.append(val_precision)
    val_recalls.append(val_recall)
    val_f1s.append(val_f1)

    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")


    # Validation metrics
    all_val_labels, all_val_preds, all_val_probs = [], [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()  
            preds = torch.argmax(outputs, dim=1).cpu().numpy()  
            all_val_labels.extend(labels.cpu().numpy())  
            all_val_preds.extend(preds) 
            all_val_probs.extend(probs)  

    # Best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improvement_epochs = 0 
        torch.save(model.state_dict(), save_path)
        print("Model saved!")
    else:
        no_improvement_epochs += 1
        print(f"No improvement for {no_improvement_epochs} epoch.")
    
    # Early Stopping
    if no_improvement_epochs >= patience:
        print(f"Early stopping triggered after {epoch + 1} epochs")
        break

    # update learning rate scheduler
    scheduler.step()

    print(f"Current Learning Rate: {scheduler.get_last_lr()[0]}")

print("Training completed.")


# Plotting metrics
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
epochs = range(1, len(train_accuracies) + 1)

# Accuracy Plot
axes[0, 0].plot(epochs, train_accuracies, label='Train Accuracy', marker='o')
axes[0, 0].plot(epochs, val_accuracies, label='Validation Accuracy', marker='o')
axes[0, 0].set_title('Accuracy Over Epochs')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid()

# F1 Score Plot
axes[0, 1].plot(epochs, train_f1s, label='Train F1 Score', marker='o')
axes[0, 1].plot(epochs, val_f1s, label='Validation F1 Score', marker='o')
axes[0, 1].set_title('F1 Score Over Epochs')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('F1 Score')
axes[0, 1].legend()
axes[0, 1].grid()

# Precision Plot
axes[1, 0].plot(epochs, train_precisions, label='Train Precision', marker='o')
axes[1, 0].plot(epochs, val_precisions, label='Validation Precision', marker='o')
axes[1, 0].set_title('Precision Over Epochs')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid()

# Recall Plot
axes[1, 1].plot(epochs, train_recalls, label='Train Recall', marker='o')
axes[1, 1].plot(epochs, val_recalls, label='Validation Recall', marker='o')
axes[1, 1].set_title('Recall Over Epochs')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].legend()
axes[1, 1].grid()

# save the figure
fig.tight_layout()
job_id = sys.argv[1] # Get job id
plt.savefig(f'/home/wooju.chung/TMA/result_tma/metrics_over_epochs_{job_id}.svg', format='svg')
plt.close()

# Combined confusion matrix and ROC curve plot
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Train set Confusion matrix
conf_matrix_train = confusion_matrix(all_train_labels, all_train_preds)
sns.heatmap(conf_matrix_train, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset_bfsplit.classes, yticklabels=train_dataset_bfsplit.classes, ax=axes[0, 0])
axes[0, 0].set_title('Train Confusion Matrix')
axes[0, 0].set_xlabel('Predicted Label')
axes[0, 0].set_ylabel('True Label')

# Train set ROC curve
fpr_train, tpr_train, roc_auc_train = {}, {}, {}
all_train_probs = np.array(all_train_probs) 

for i in range(len(train_dataset_bfsplit.classes)): 
    if i == 0:  # benign
        fpr_train[i], tpr_train[i], _ = roc_curve(1 - np.array(all_train_labels), all_train_probs[:, i])
    else:  # tumor
        fpr_train[i], tpr_train[i], _ = roc_curve(np.array(all_train_labels), all_train_probs[:, i])
    roc_auc_train[i] = auc(fpr_train[i], tpr_train[i])
    axes[0, 1].plot(fpr_train[i], tpr_train[i], label=f"{train_dataset_bfsplit.classes[i]} (AUC = {roc_auc_train[i]:.2f})")

axes[0, 1].plot([0, 1], [0, 1], linestyle='--', color='gray')
axes[0, 1].set_title('Train ROC Curve')
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].legend(loc='lower right')

# Val set confusion matrix

conf_matrix_val = confusion_matrix(all_val_labels, all_val_preds)
sns.heatmap(conf_matrix_val, annot=True, fmt='d', cmap='Blues', xticklabels=val_dataset.classes, yticklabels=val_dataset.classes, ax=axes[1, 0])
axes[1, 0].set_title('Validation Confusion Matrix')
axes[1, 0].set_xlabel('Predicted Label')
axes[1, 0].set_ylabel('True Label')

# Val set ROC Curve
fpr_val, tpr_val, roc_auc_val = {}, {}, {}
all_val_probs = np.array(all_val_probs)

for i in range(len(train_dataset_bfsplit.classes)): 
    if i == 0:  # benign
        fpr_val[i], tpr_val[i], _ = roc_curve(1 - np.array(all_val_labels), all_val_probs[:, i])
    else:  # tumor
        fpr_val[i], tpr_val[i], _ = roc_curve(all_val_labels, all_val_probs[:, i])
    roc_auc_val[i] = auc(fpr_val[i], tpr_val[i])
    axes[1, 1].plot(fpr_val[i], tpr_val[i], label=f"{train_dataset_bfsplit.classes[i]} (AUC = {roc_auc_val[i]:.2f})")

axes[1, 1].plot([0, 1], [0, 1], linestyle='--', color='gray')
axes[1, 1].set_title('Validation ROC Curve')
axes[1, 1].set_xlabel('False Positive Rate')
axes[1, 1].set_ylabel('True Positive Rate')
axes[1, 1].legend(loc='lower right')

fig.tight_layout()
plt.savefig(f'/home/wooju.chung/TMA/result_tma/confusion_roc_{job_id}.svg', format='svg')
plt.close()

# Test evaluation
model.eval()
test_loss = 0.0
all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)

        # Softmax
        preds = torch.argmax(outputs, dim=1) 
        probs = torch.softmax(outputs, dim=1).cpu().numpy() 


        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs) 

test_loss /= len(test_loader.dataset)
print(f"Test Loss: {test_loss:.4f}")

# calcuate accuracy 
accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"Accuracy: {accuracy:.4f}")

# Precision, Recall, F1 Score
precision = precision_score(all_labels, all_preds, pos_label=1)
recall = recall_score(all_labels, all_preds, pos_label=1)
f1 = f1_score(all_labels, all_preds, pos_label=1)
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# test set Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset_bfsplit.classes, yticklabels=train_dataset_bfsplit.classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Confusion Matrix testset
plt.savefig(f'/home/wooju.chung/TMA/result_tma/confusion_matrix_{job_id}.svg', format='svg')  
plt.close()

# Test ROC Curve
plt.figure(figsize=(8, 6))
fpr_test, tpr_test, roc_auc_test = {}, {}, {}
all_probs = np.array(all_probs)  

for i in range(len(train_dataset_bfsplit.classes)): 
    if i == 0:  # for benign class
        fpr_test[i], tpr_test[i], _ = roc_curve(1 - np.array(all_labels), all_probs[:, i])
    else:  # for tumor class
        fpr_test[i], tpr_test[i], _ = roc_curve(np.array(all_labels), all_probs[:, i])
    roc_auc_test[i] = auc(fpr_test[i], tpr_test[i])
    plt.plot(fpr_test[i], tpr_test[i], label=f"{train_dataset_bfsplit.classes[i]} (AUC = {roc_auc_test[i]:.2f})")

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('Test ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')

plt.savefig(f'/home/wooju.chung/TMA/result_tma/roc_curve_test_{job_id}.svg', format='svg')  
plt.close()

# Train Loss, Val Loss
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Loss Over Epochs')
plt.legend()
plt.grid()
# save as a svg file
plt.savefig(f'/home/wooju.chung/TMA/result_tma/train_val_loss_{job_id}.svg', format='svg') 
