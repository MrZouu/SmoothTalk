import os
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models, datasets
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm # Afficher entrainement en temps reel
import matplotlib.pyplot as plt # Dessiner graphes
from sklearn.metrics import precision_recall_fscore_support

# CUDA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("Using the ASL DATASET")

print(torch.cuda.is_available()) # Si cuda disponible / bien installé
print(torch.cuda.device_count()) # Quantité de GPU ( 0 si non fonctionnel )
torch.cuda.get_device_name() # Nom de la carte graphique

# Pré-Traitement

# Définir la transformation des données
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))  # Normalisation des valeurs de pixel centré autour de 0 (-1 ; 1)
])

# Charger données

# Chemins des répertoires de division - Train et Test strictement séparés
train_dataset_path = 'Train_Dataset'
test_dataset_path = 'Test_Dataset'

# Charger le dataset ASL Alphabet
asl_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=transform)

# Créer une fonction pour charger les images depuis un répertoire
def load_images_from_directory(directory):
    images = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg"):  # Assurez-vous de spécifier la bonne extension d'image
                image_path = os.path.join(root, file)
                images.append((image_path, asl_dataset.class_to_idx[root.split('\\')[-1]]))
    return images

# Charger les ensembles d'entraînement et de test
train_dataset = load_images_from_directory(train_dataset_path)
test_dataset = load_images_from_directory(test_dataset_path)

# Appliquer les transformations aux ensembles d'entraînement et de test
train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root=test_dataset_path, transform=transform)

# Définir les chargeurs de données
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Resnet

class ResNet50(nn.Module):
    def __init__(self, num_classes, weights=None):
        super(ResNet50, self).__init__()
        # Charger le modèle pré-entraîné ResNet50
        self.resnet = resnet50(pretrained=True if weights is None else False)

        # Extraction des couches sauf la couche de classification finale (dernière couche)
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])

        # Nouvelle couche de classification adaptée au nombre de classes (26 lettres)
        self.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Instancier le modèle ResNet50, la fonction de coût et l'optimiseur
num_classes = 26  # Nombre de classes correspondant aux lettres de l'alphabet
# model = ResNet50(num_classes, weights=None).to(device)
model = ResNet50(num_classes, weights=ResNet50_Weights.DEFAULT).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 13
# Préparation des métriques
train_losses = []
train_accuracies = []
test_accuracies = []
class_precisions = []
class_recalls = []
class_f1_scores = []

# Entrainement

# Entraîner le modèle
# Initialiser les listes pour stocker les statistiques d'entraînement
train_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False, position=0, dynamic_ncols=True)

    for batch_idx, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        progress_bar.set_postfix(loss=f'{running_loss / (batch_idx + 1):.3f}', accuracy=f'{100 * correct / total:.2f}%')

    # Imprimer la statistique de l'ensemble d'entraînement à la fin de l'époque
    average_loss = running_loss / len(train_loader)
    accuracy = correct / total
    train_losses.append(average_loss)
    train_accuracies.append(accuracy)
    progress_bar.set_postfix(loss=f'{average_loss:.3f}', accuracy=f'{100 * accuracy:.2f}%')
    progress_bar.close()  # Fermer la barre de progression à la fin de l'époque

    # Tester le modèle sur les données de test après chaque époque
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predicted = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())

    # Enregistrer la précision des tests
    test_accuracy = correct / total
    test_accuracies.append(test_accuracy)

precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_predicted, average=None)
class_precisions.append(precision)
class_recalls.append(recall)
class_f1_scores.append(f1_score)

# Afficher les métriques par classe
for i, (precision, recall, f1_score) in enumerate(zip(class_precisions[-1], class_recalls[-1], class_f1_scores[-1])):
    print(f"Class {i}: Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1_score:.4f}")

# Graphes

# Tracer les courbes d'entraînement
plt.figure(figsize=(12, 4))

# Courbe de perte
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()

# Courbe d'Accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy', marker='o')
plt.plot(test_accuracies, label='Test Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy Over Epochs')
plt.legend()

plt.tight_layout()
plt.show()

# Enregistrement du modèle

# Enregistrer le modèle à la fin de l'entraînement
torch.save(model.state_dict(), 'pytorch_model.pth')
print("Le modèle a été enregistré.")
