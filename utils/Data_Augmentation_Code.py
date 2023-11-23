import Augmentor
import os
import shutil
import cv2
import numpy as np

# Augmentation des images d'un dataset

def augmenter_contraste(image_path, output_path, counter):
    # Vérifier fichier
    if not os.path.isfile(image_path):
        print(f"Le fichier {image_path} n'existe pas.")
        return

    # Charger image avec OpenCV
    image = cv2.imread(image_path)

    # Vérifier image lue
    if image is None:
        print(f"Impossible de lire l'image à partir de {image_path}.")
        return

    # augmentation de contraste
    alpha = np.random.uniform(0.8, 1.2)
    beta = np.random.uniform(-20, 20)
    image_augmented = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Extraire le nom de l'image sans le chemin
    image_name = os.path.basename(image_path)

    # Construire le chemin de sortie
    output_image_path = os.path.join(output_path, f"{image_name[:-4]}_{counter}.jpg")

    # Enregistrer l'image augmentée
    cv2.imwrite(output_image_path, image_augmented)


def augmenter_dataset(input_path, output_path, nombre_images_augmentees):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for lettre in alphabet:
        lettre_input_path = os.path.join(input_path, lettre)
        lettre_output_path = os.path.join(output_path, lettre)
        os.makedirs(lettre_output_path, exist_ok=True)

        # Augmentation des images
        for existing_image in os.listdir(lettre_input_path):
            existing_image_path = os.path.join(lettre_input_path, existing_image)

            # Appliquer l'augmentation de contraste
            augmenter_contraste(existing_image_path, lettre_output_path, 0)

        # Créer des images supplémentaires si nécessaire
        pipeline = Augmentor.Pipeline(lettre_input_path)
        pipeline.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
        pipeline.flip_left_right(probability=0.5)
        pipeline.flip_top_bottom(probability=0.5)
        pipeline.zoom_random(probability=0.5, percentage_area=0.8)

        counter = 1
        for operation in pipeline.operations:
            if isinstance(operation, Augmentor.Operations.Lambda):
                operation.function.keywords["counter"] = counter
                counter += 1

        # Ajout d'une augmentation de contraste
        pipeline.add_operation(ContrasteOperation(probability=1.0, counter=counter))

        pipeline.sample(nombre_images_augmentees)


# Utilisation
input_dataset_path = r"vers/chemin/dataset"
output_augmented_path = r"vers/chemin/sortie"
nombre_images_augmentees = 550  # Nombres images crées

augmenter_dataset(input_dataset_path, output_augmented_path, nombre_images_augmentees)
