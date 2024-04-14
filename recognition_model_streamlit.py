# Importation des bibliothèques nécessaires
import cv2                              # OpenCV pour la manipulation d'images
import mediapipe as mp                  # Mediapipe pour la détection des mains
import torch                            # PyTorch pour l'apprentissage automatique
import torchvision.transforms as transforms  # Transformations d'images pour PyTorch
import torch.nn.functional as F         # Fonctions de perte et d'activation pour PyTorch
import torch.nn as nn                   # Réseaux de neurones pour PyTorch
from torchvision.models import resnet50 # Modèle ResNet50 pré-entraîné dans torchvision
import numpy as np                      # Manipulation de tableaux numériques avec NumPy
import ssl                              # Pour gérer les problèmes de certificat SSL

# Désactivation de la vérification SSL
ssl._create_default_https_context = ssl._create_unverified_context

# Définition d'un modèle de réseau de neurones basé sur ResNet50 pour la classification
class ResNet50(nn.Module):
    def __init__(self, num_classes, weights=None):
        super(ResNet50, self).__init__()
        # Chargement du modèle ResNet50 pré-entraîné
        self.resnet = resnet50(weights=True if weights is None else False)
        # Extraction des couches sauf la couche de classification finale
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        # Ajout d'une nouvelle couche de classification adaptée au nombre de classes (26 lettres)
        self.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Détection du dispositif disponible pour l'exécution (GPU s'il est disponible, sinon CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Instanciation du modèle ResNet50 pour la classification de 26 classes (lettres de l'alphabet des signes)
model = ResNet50(26, weights="T3pytorch_model.pth").to(device)

# Met le modèle en mode évaluation (pas de mise à jour des poids)
model.eval()

# Chargement du détecteur de main de Mediapipe
mp_hands = mp.solutions.hands
hands_model = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Fonction pour détecter et traduire la langue des signes dans une image
def translate_sign_language(img):
    # Conversion de l'image de BGR à RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Détection des mains dans l'image
    hands_results = hands_model.process(imgRGB)

    # Si des mains sont détectées
    if hands_results.multi_hand_landmarks:
        for handLms in hands_results.multi_hand_landmarks:
            # Extraction des landmarks de la main détectée
            landmarks = handLms.landmark
            x_values = [landmark.x for landmark in landmarks]
            y_values = [landmark.y for landmark in landmarks]
            min_x, max_x = min(x_values), max(x_values)
            min_y, max_y = min(y_values), max(y_values)

            # Définition d'une marge autour de la main détectée
            margin = 30
            min_x = max(0, int(min_x * img.shape[1]) - margin)
            min_y = max(0, int(min_y * img.shape[0]) - margin)
            max_x = min(img.shape[1], int(max_x * img.shape[1]) + margin)
            max_y = min(img.shape[0], int(max_y * img.shape[0]) + margin)

            # Extraction de la région d'intérêt (ROI) autour de la main détectée
            bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
            roi = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            roi_resized = cv2.resize(roi, (224, 224))

            # Prétraitement de l'image pour l'entrée du modèle
            img_tensor = transforms.ToTensor()(roi_resized).unsqueeze(0).to(device)

            # Prédiction de la lettre de la langue des signes à partir de l'image
            with torch.no_grad():
                model_output = model(img_tensor)
            predicted_class = torch.argmax(F.softmax(model_output, dim=1), dim=1).item()
            predicted_letter = chr(ord('A') + predicted_class)

            # Affichage de la lettre prédite sur l'image
            cv2.putText(img, predicted_letter, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Retour de l'image avec les traductions affichées
    return img
