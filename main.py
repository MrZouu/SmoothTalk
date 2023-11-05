from keras import layers, models
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Chemin vers les fichiers Train et Test
train_dir = r"C:\Users\lor3n\Desktop\Dataset_Train"
test_dir = r"C:\Users\lor3n\Desktop\Dataset_Test"


# Fonction pour charger les données à partir du dossier (car dossiers imbriqués)
def load_data(directory):
    data = []
    labels = []
    class_names = os.listdir(directory)

    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        class_label = class_names.index(class_name)

        for filename in os.listdir(class_dir):
            if filename.endswith('.jpg'):
                img = cv2.imread(os.path.join(class_dir, filename))
                img = cv2.resize(img, (224, 224))  # Redimensionner l'image
                img = img / 255.0
                data.append(img)
                labels.append(class_label)

    data = np.array(data)
    labels = np.array(labels)

    return data, labels


# Charger les données d'entraînement et de test
train_data, train_labels = load_data(train_dir)
test_data, test_labels = load_data(test_dir)

# Divisez les données d'entraînement en données d'entraînement et de validation
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2,
                                                                  random_state=42)
# test-size = 0,2 veut dire que 20% des images entrainement iront pour la validation

# Assurez-vous que les étiquettes sont au format "one-hot encoding"
train_labels = to_categorical(train_labels, num_classes=3)  # 3 classes (A, B, C)
val_labels = to_categorical(val_labels, num_classes=3)
test_labels = to_categorical(test_labels, num_classes=3)


# TEST
print("Forme des données d'entraînement :", train_data.shape)
print("Forme des étiquettes d'entraînement :", train_labels.shape)
print("Forme des données de validation :", val_data.shape)
print("Forme des étiquettes de validation :", val_labels.shape)
print("Forme des données de test :", test_data.shape)
print("Forme des étiquettes de test :", test_labels.shape)


# Créez un modèle séquentiel #############
model = models.Sequential()

# Couches de convolution
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
# 32 filtres en (matrice),relu (Rectified Linear Unit) pour garder que valeurs positives,images 224x224 en 3 canaux(RGB)
model.add(layers.MaxPooling2D((2, 2)))
# Couche pooling pour réduire la taille de l'image
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# Ici on a 64 filtres, chaque filtre indépendant et détecte une caractéristique
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Couche de sortie
model.add(layers.Flatten())  # Transforme les caractéristiques en un seul vecteur
model.add(layers.Dense(64, activation='relu'))  # couche dense 64 neurones
model.add(layers.Dense(3, activation='softmax'))  # 3 classes (A, B, C), softmax pour calculer la probabilité

# Compilation du modèle
model.compile(optimizer='adam',  # adam est algo optimisation ajustement poids du modele
              loss='categorical_crossentropy',  # Fonction de perte, mesure erreur prédictions.
              metrics=['accuracy'])  # métrique évaluation de performance

# Entraînement du modèle (données d'entraînement)
# train_data, train_labels, val_data et val_labels
history = model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(val_data, val_labels))
# train data = données entrainement, train_labels = catégories, epoch = nbr passage sur données
# batch size = division en lots images, validation_data=(val_data, val_labels) = évaluation à fin d'époque
# val_data = images réservées pour ensemble de validation.
# La validation sert à reperer si le modele surapprend (overfitting).

# Evaluer les données de test permet en revanche de voir les performances sur des données jamais vues.

# Évaluation du modèle (données de test)
# Supposons test_data et test_labels
test_loss, test_accuracy = model.evaluate(test_data, test_labels)
# test-data = données de test, jamais vues avant. test-labels = classes
print(f"Perte sur les données de test : {test_loss}")  # Perte / erreur calculée
print(f"Précision sur les données de test : {test_accuracy}")  # images correctement classées

model.save('modele.h5')  # On sauvegarde le modèle ici
