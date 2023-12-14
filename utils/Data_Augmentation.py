import os
import cv2
import random
import imgaug.augmenters as iaa

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

def process_images(input_folder, output_folder, target_count):
    for root, dirs, files in os.walk(input_folder):
        for subdir in dirs:
            class_folder = os.path.join(output_folder, subdir)
            os.makedirs(class_folder, exist_ok=True)
            images = os.listdir(os.path.join(input_folder, subdir))

            # Calculer le nombre d'images à ajouter pour atteindre target_count
            images_to_add = max(target_count - len(images), 0)

            # Dupliquer les images si nécessaire pour atteindre target_count
            selected_images = random.choices(images, k=images_to_add) + images

            for i, image_name in enumerate(selected_images):
                original_path = os.path.join(input_folder, subdir, image_name)
                original_image = cv2.imread(original_path)

                # Data augmentation
                augmented_image = augment_image(original_image)

                # Renommage et sauvegarde
                new_name = f"{subdir}{i + 1}.jpg"
                new_path = os.path.join(class_folder, new_name)
                cv2.imwrite(new_path, augmented_image)

if __name__ == "__main__":
    input_folder = r"C:\Users\lor3n\Desktop\data2"
    output_folder = r"C:\Users\lor3n\Desktop\data2Aug"
    target_count = 3000

    process_images(input_folder, output_folder, target_count)
