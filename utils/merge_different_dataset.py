import os
from shutil import copytree
# Fusionner plusieurs datasets en un seul
def fusionner_datasets(dataset_A_path, dataset_B_path, dataset_C_path, output_path):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for lettre in alphabet:
        lettre_path_A = os.path.join(dataset_A_path, lettre)
        lettre_path_B = os.path.join(dataset_B_path, lettre)
        lettre_path_C = os.path.join(dataset_C_path, lettre)

        if os.path.isdir(lettre_path_A) and os.path.isdir(lettre_path_B) and os.path.isdir(lettre_path_C):
            output_lettre_path = os.path.join(output_path, lettre)
            os.makedirs(output_lettre_path, exist_ok=True)

            copytree(lettre_path_A, output_lettre_path, dirs_exist_ok=True)
            copytree(lettre_path_B, output_lettre_path, dirs_exist_ok=True)
            copytree(lettre_path_C, output_lettre_path, dirs_exist_ok=True)

# Utilisation
dataset_A_path = r"chemin/vers/dataset_A"
dataset_B_path = r"chemin/vers/dataset_B"
dataset_C_path = r"chemin/vers/dataset_B"

output_dataset_path = r"chemin/vers/sortie"

fusionner_datasets(dataset_A_path, dataset_B_path, dataset_C_path, output_dataset_path)