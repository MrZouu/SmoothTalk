import os
import cv2
import random


def augment_image(image):
    # Fonction de data augmentation
    # Vous pouvez ajouter ou modifier les transformations selon vos besoins
    image = cv2.GaussianBlur(image, (5, 5), 0)  # Modification de flou
    image = cv2.flip(image, 1)  # Miroir horizontal
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Modification de luminosité et contraste
    alpha = 1.0 + random.uniform(-0.2, 0.2)
    beta = random.uniform(-20, 20)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image


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
    input_folder = r"C:\Users\lor3n\Desktop\DATASET1"
    output_folder = r"C:\Users\lor3n\Desktop\data"
    target_count = 3000

    process_images(input_folder, output_folder, target_count)
