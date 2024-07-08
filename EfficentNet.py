import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import os
from tqdm import tqdm
from PIL import Image


# Definizione del dataset personalizzato per gli spettrogrammi
class SpectrogramDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self.class_to_idx = {}

        # Carica i dati e le etichette
        for class_idx, class_name in enumerate(sorted(os.listdir(data_dir))):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                self.class_to_idx[class_name] = class_idx
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.data.append(img_path)
                        self.labels.append(class_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]

        # Carica l'immagine
        image = Image.open(img_path).convert('RGB')

        # Applica le trasformazioni se specificate
        if self.transform:
            image = self.transform(image)

        return image, label


# Iperparametri
num_epochs = 25
batch_size = 32
learning_rate = 0.001
patience = 5
num_classes = 5  # Modifica in base al tuo dataset
data_path = "Alfabeto1_Multiclasse_NoAug"

# Preparazione del dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = SpectrogramDataset(os.path.join(data_path, 'train'), transform=transform)
val_dataset = SpectrogramDataset(os.path.join(data_path, 'val'), transform=transform)

print(f"Numero di campioni nel training set: {len(train_dataset)}")
print(f"Numero di campioni nel validation set: {len(val_dataset)}")

if len(train_dataset) == 0 or len(val_dataset) == 0:
    raise ValueError("Uno o entrambi i dataset sono vuoti. Verifica i percorsi dei dati.")

num_classes = len(train_dataset.class_to_idx)
print(f"Numero di classi: {num_classes}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Carica il modello EfficientNet pre-addestrato
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Funzione per calcolare le metriche
def calculate_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return precision, recall, f1


# Inizializzazione delle variabili per il salvataggio
best_f1 = 0
best_epoch = 0
epochs_no_improve = 0
metrics = []

# Ciclo di addestramento
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_pred = []
    train_true = []

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_pred.extend(predicted.cpu().numpy())
        train_true.extend(labels.cpu().numpy())

    train_loss /= len(train_loader)
    train_precision, train_recall, train_f1 = calculate_metrics(train_true, train_pred)

    # Validazione
    model.eval()
    val_loss = 0
    val_pred = []
    val_true = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_pred.extend(predicted.cpu().numpy())
            val_true.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    val_precision, val_recall, val_f1 = calculate_metrics(val_true, val_pred)

    # Salvataggio dei pesi dell'epoca corrente
    torch.save(model.state_dict(), f'model_epoch_{epoch + 1}.pth')

    # Aggiornamento del miglior modello
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_epoch = epoch
        torch.save(model.state_dict(), 'best_model.pth')
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    # Early stopping
    if epochs_no_improve == patience:
        print(f"Early stopping triggered after {epoch + 1} epochs")
        break

    # Salvataggio delle metriche
    metrics.append({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'train_precision': train_precision,
        'train_recall': train_recall,
        'train_f1': train_f1,
        'val_loss': val_loss,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_f1': val_f1
    })

# Salvataggio delle metriche in un file CSV
pd.DataFrame(metrics).to_csv('training_metrics.csv', index=False)

print(f"Addestramento completato. Miglior epoca: {best_epoch + 1}")