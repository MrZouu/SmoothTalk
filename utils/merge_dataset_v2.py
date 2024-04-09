import os
import random
import shutil

def mélanger_dataset(chemin_dataset1, chemin_dataset2, chemin_dataset3, chemin_dataset4, chemin_final, quantités_par_lettre):
    lettres_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Créer le répertoire final s'il n'existe pas
    if not os.path.exists(chemin_final):
        os.makedirs(chemin_final)

    # Parcourir chaque lettre de l'alphabet
    for lettre, quantité_par_lettre in zip(lettres_alphabet, quantités_par_lettre):
        images_dataset1 = []
        images_dataset2 = []
        images_dataset3 = []
        images_dataset4 = []

        # Trouver les fichiers correspondants à la lettre dans chaque dataset
        for chemin_dataset, images_dataset in zip([chemin_dataset1, chemin_dataset2, chemin_dataset3, chemin_dataset4],
                                                 [images_dataset1, images_dataset2, images_dataset3, images_dataset4]):
            chemin_lettre = os.path.join(chemin_dataset, lettre)
            if os.path.exists(chemin_lettre):
                images_dataset.extend([os.path.join(chemin_lettre, fichier) for fichier in os.listdir(chemin_lettre)])

        # Sélectionner le nombre spécifié d'images pour cette lettre
        images_mélangées = random.sample(images_dataset1 + images_dataset2 + images_dataset3+ images_dataset4, quantité_par_lettre)

        # Créer le répertoire de la lettre dans le répertoire final s'il n'existe pas
        chemin_final_lettre = os.path.join(chemin_final, lettre)
        if not os.path.exists(chemin_final_lettre):
            os.makedirs(chemin_final_lettre)

        # Copier les images mises en commun vers le répertoire final
        for image in images_mélangées:
            shutil.copy(image, os.path.join(chemin_final_lettre, os.path.basename(image)))

# Utilisation du script
chemin_dataset1 = r"C:\Users\Lorenzo\Desktop\ASL1"
chemin_dataset2 = r"C:\Users\Lorenzo\Desktop\ASL2"
chemin_dataset3 = r"C:\Users\Lorenzo\Desktop\ASL3"
chemin_dataset4 = r"C:\Users\Lorenzo\Desktop\ASL4"
chemin_final = r"C:\Users\Lorenzo\Desktop\DataMix4"

# Liste des quantités par lettre, ajustez selon vos besoins
quantités_par_lettre = [2500,  # A
                        1900,   # B
                        1900,   # C
                        1900,   # D
                        2500,  # E
                        2500,   # F
                        2500,   # G
                        1900,   # H
                        2500,  # I
                        2500,  # J
                        2500,  # K
                        1900,   # L
                        2500,  # M
                        1900,   # N
                        2500,   # O
                        2500,  # P
                        2500,  # Q
                        1900,   # R
                        2500,  # S
                        2500,  # T
                        1900,   # U
                        2500,  # V
                        1900,   # W
                        1900,   # X
                        1900,   # Y
                        2500]   # Z

mélanger_dataset(chemin_dataset1, chemin_dataset2, chemin_dataset3, chemin_dataset4, chemin_final, quantités_par_lettre)