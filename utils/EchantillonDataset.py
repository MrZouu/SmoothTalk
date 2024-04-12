import os
import shutil
import random

# Chemin vers le répertoire du dataset d'origine
chemin_du_dataset = r"C:\Users\Lorenzo\Desktop\T7\MixCustomKaggle5"

# Chemin vers le nouveau répertoire à créer
chemin_nouveau_dataset = r"C:\Users\Lorenzo\Desktop\T7_Echantillon"

# Nombre d'images à sélectionner aléatoirement par classe
nombre_images_par_classe = 100

# Vérifier si le répertoire du nouveau dataset existe, sinon le créer
if not os.path.exists(chemin_nouveau_dataset):
    os.makedirs(chemin_nouveau_dataset)

# Parcourir toutes les classes du dataset d'origine
for classe in os.listdir(chemin_du_dataset):
    chemin_classe_origine = os.path.join(chemin_du_dataset, classe)
    chemin_classe_destination = os.path.join(chemin_nouveau_dataset, classe)

    # Vérifier si le répertoire de la classe dans le nouveau dataset existe, sinon le créer
    if not os.path.exists(chemin_classe_destination):
        os.makedirs(chemin_classe_destination)

    # Sélectionner aléatoirement des images dans la classe d'origine
    images_origine = os.listdir(chemin_classe_origine)
    images_a_copier = random.sample(images_origine, min(nombre_images_par_classe, len(images_origine)))

    # Copier les images sélectionnées vers le nouveau répertoire
    for image in images_a_copier:
        chemin_image_origine = os.path.join(chemin_classe_origine, image)
        chemin_image_destination = os.path.join(chemin_classe_destination, image)
        shutil.copyfile(chemin_image_origine, chemin_image_destination)

print("Création du nouvel échantillon terminée.")
