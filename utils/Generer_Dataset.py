import cv2
import os

# Spécifier le répertoire de sauvegarde du dataset
dataset_dir = r"C:\Users\lor3n\Desktop\data2"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Liste des lettres de l'alphabet
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

current_letter_index = 0

# Configurer la capture vidéo depuis la webcam
cap = cv2.VideoCapture(0)  # 0 indique la première webcam, ajustez si vous en avez plusieurs

while True:
    # Lire le cadre vidéo
    ret, frame = cap.read()

    # Afficher la lettre actuelle à l'écran
    cv2.putText(frame, f'Lettre actuelle : {alphabet[current_letter_index]}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Afficher le cadre vidéo
    cv2.imshow('Webcam', frame)

    # Attendre l'appui sur une touche
    key = cv2.waitKey(1) & 0xFF

    # Si la touche 'q' est enfoncée, quitter la boucle
    if key == ord('q'):
        break

    # Si la touche 'c' est enfoncée, capturer une image
    elif key == ord('c'):
        # Générer le chemin du sous-dossier correspondant à la lettre
        letter_dir = os.path.join(dataset_dir, alphabet[current_letter_index])
        if not os.path.exists(letter_dir):
            os.makedirs(letter_dir)

        # Générer le chemin de sauvegarde pour l'image
        save_path = os.path.join(letter_dir, f'{alphabet[current_letter_index]}{len(os.listdir(letter_dir)) + 1}.jpg')

        # Enregistrer l'image
        cv2.imwrite(save_path, frame)
        print(f"Image enregistrée : {save_path}")

    # Si la touche 'n' est enfoncée, passer à la lettre suivante
    elif key == ord('n'):
        current_letter_index = (current_letter_index + 1) % len(alphabet)
        print(f"Passer à la lettre suivante : {alphabet[current_letter_index]}")

# Libérer la capture vidéo et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()