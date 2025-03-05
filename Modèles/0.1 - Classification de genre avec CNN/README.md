# Objectif
Ce notebook se concentre sur la prédiction du genre (classification binaire) à partir d'images de visages. Il inclut le prétraitement des données, la préparation des ensembles d'entraînement, de validation et de test, ainsi que la définition d'un pipeline de données pour l'entraînement d'un modèle de deep learning.


# 1. Prétraitement des données
- **Organisation des données** :
  - Les images sont chargées à partir du dataset UTKFace, où le nom de fichier contient le genre.
  - Les données sont divisées en ensembles d'entraînement (60 %), de validation (20 %) et de test (20 %) à l'aide de `train_test_split`.
  - Les images sont organisées dans des dossiers structurés par genre (`male`, `female`) pour faciliter leur utilisation avec `image_dataset_from_directory`.

- **Fonction `preprocess_utkface_dataset`** :
  - Charge les images et extrait les labels de genre à partir des noms de fichiers.
  - Crée des dossiers pour les ensembles d'entraînement, de validation et de test, et copie les images dans les dossiers appropriés.

# 2. Chargement des données avec `image_dataset_from_directory`
- **Utilisation de `image_dataset_from_directory`** :
  - Charge les images et les labels directement à partir des dossiers structurés.
  - Redimensionne les images à une taille fixe (128x128 pixels).
  - Définit les classes comme `female` et `male`.

# 3. Prétraitement des images
- **Fonction `preprocess_utkface`** :
  - **Pour l'entraînement** :
    - Applique un recadrage aléatoire (`random_crop`) et un retournement horizontal aléatoire (`random_flip_left_right`).
    - Normalise les valeurs des pixels entre 0 et 1.
  - **Pour la validation et le test** :
    - Applique un recadrage centré (`crop_to_bounding_box`) pour s'assurer que les visages sont bien centrés.
    - Redimensionne et normalise les images.

# 4. Pipeline de données avec `tf.data`
- **Shuffle et batch** :
  - Les données d'entraînement sont mélangées (`shuffle`) et divisées en lots (`batch`) pour l'entraînement.
  - Les données de validation et de test sont également divisées en lots.
- **Optimisation des performances** :
  - Utilisation de `cache()` et `prefetch()` pour améliorer les performances lors de l'entraînement.
  - 
## Alternatives possibles

#### 3. **Pipeline de données**
- **Batch size** :
  - Ajuster la taille des lots (`batch_size`) en fonction de la mémoire disponible et des performances du modèle.

