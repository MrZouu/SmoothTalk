import os
import cv2
import random
import imgaug.augmenters as iaa
import shutil

def augment_image(image):
    # Définir la séquence de transformations aléatoires
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # Flip horizontal avec une probabilité de 50%
        iaa.Sometimes(0.5, iaa.Crop(percent=(0, 0.2))),  # Crop avec une probabilité de 50%
        iaa.GaussianBlur(sigma=(0, 1.0)),  # Flou gaussien
        iaa.AddToHueAndSaturation(value=(-10, 10)),  # Modification de la teinte et de la saturation
        iaa.Add((-10, 10), per_channel=0.5),  # Ajouter un nombre aléatoire à chaque pixel pour la luminosité
        iaa.Multiply((0.9, 1.1), per_channel=0.5),  # Multiplier chaque pixel par un nombre aléatoire pour le contraste
    ])

    # Appliquer les transformations à l'image
    augmented_image = seq.augment_image(image)

    return augmented_image

def process_images(input_folder, output_folder, target_counts):
    for root, dirs, files in os.walk(input_folder):
        for subdir in dirs:
            class_folder = os.path.join(output_folder, subdir)
            os.makedirs(class_folder, exist_ok=True)
            images = os.listdir(os.path.join(input_folder, subdir))

            # Copier les images originales dans le dossier de destination
            for image_name in images:
                original_path = os.path.join(input_folder, subdir, image_name)
                destination_path = os.path.join(class_folder, image_name)
                shutil.copy(original_path, destination_path)

            # Calculer le nombre d'images à ajouter pour atteindre target_counts[subdir]
            images_to_add = max(target_counts[subdir] - len(images), 0)

            # Dupliquer les images si nécessaire pour atteindre target_counts[subdir]
            selected_images = random.choices(images, k=images_to_add) + images

            for i, image_name in enumerate(selected_images):
                original_path = os.path.join(input_folder, subdir, image_name)
                original_image = cv2.imread(original_path)

                # Data augmentation
                augmented_image = augment_image(original_image)

                # Rogner l'image pour obtenir un format carré
                h, w, _ = augmented_image.shape
                size = min(h, w)
                cropped_image = augmented_image[:size, :size]

                # Renommage et sauvegarde
                new_name = f"{image_name.split('.')[0]}_augmented{i + 1}.jpg"
                new_path = os.path.join(class_folder, new_name)
                cv2.imwrite(new_path, cropped_image)

if __name__ == "__main__":
    input_folder = r"C:\Users\Lorenzo\Desktop\DataMix4"
    output_folder = r"C:\Users\Lorenzo\Desktop\Dataugment4"

    # Liste des quantités cibles par lettre, ajustez selon vos besoins
    target_counts = {
        "A": 4000,
        "B": 4000,
        "C": 3500,
        "D": 3500,
        "E": 4000,
        "F": 3500,
        "G": 3500,
        "H": 3500,
        "I": 4000,
        "J": 3500,
        "K": 3500,
        "L": 3500,
        "M": 4000,
        "N": 3500,
        "O": 4000,
        "P": 4000,
        "Q": 4000,
        "R": 4000,
        "S": 4000,
        "T": 4000,
        "U": 3500,
        "V": 4000,
        "W": 3500,
        "X": 3500,
        "Y": 3500,
        "Z": 4000,
        # Ajoutez d'autres lettres avec leurs quantités cibles ici
    }

    process_images(input_folder, output_folder, target_counts)