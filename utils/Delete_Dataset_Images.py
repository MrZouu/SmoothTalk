import os
import random
# Supprimer des images d'un dataset
def supprimer_images_single_file(input_path, pourcentage_a_supprimer):
    for lettre in os.listdir(input_path):
        lettre_path = os.path.join(input_path, lettre)
        if os.path.isdir(lettre_path):
            images = os.listdir(lettre_path)
            nombre_images_a_supprimer = int(len(images) * (pourcentage_a_supprimer / 100))
            images_a_supprimer = random.sample(images, nombre_images_a_supprimer)
            for image in images_a_supprimer:
                image_path = os.path.join(lettre_path, image)
                os.remove(image_path)

# Utilisation
input_single_file_path = r"C:\Users\lor3n\Desktop\data"
pourcentage_a_supprimer = 50  # Ajuster le pourcentage selon besoins ( % enlevé )

supprimer_images_single_file(input_single_file_path, pourcentage_a_supprimer)
