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

# 5. **Architecture du modèle**
- **Couches convolutives** :
  - Le modèle utilise 4 couches convolutives avec des filtres de taille 3x3 et une activation ReLU.
  - Chaque couche convolutive est suivie d'une couche de `MaxPooling2D` pour réduire la dimension spatiale des caractéristiques.
  - La régularisation L2 est appliquée aux noyaux des couches convolutives pour éviter le surapprentissage.
- **Couches denses** :
  - Trois couches denses avec activation ReLU sont utilisées pour combiner les caractéristiques extraites.
  - Une régularisation L2 est également appliquée aux couches denses.
- **Dropout** :
  - Un taux de dropout de 0.5 est utilisé pour réduire le surapprentissage en désactivant aléatoirement 50 % des neurones pendant l'entraînement.
- **Sortie** :
  - Une couche dense avec un seul neurone et une activation sigmoïde est utilisée pour la classification binaire (homme/femme).

## Alternatives possibles

- **Batch Normalization** :
  - Ajouter des couches de normalisation par lots (`BatchNormalization`) après chaque couche dense pour stabiliser l'entraînement.

# 6. **Compilation du modèle**
- **Fonction de perte** : `binary_crossentropy` pour la classification binaire.
- **Optimiseur** : Adam avec un taux d'apprentissage de 0.0001.
- **Métriques** : Précision (`accuracy`) pour évaluer les performances du modèle.

### **Callbacks**
- **Early Stopping** :
  - Arrête l'entraînement si la perte de validation ne s'améliore pas pendant 5 époques.
  - Restaure les meilleurs poids du modèle.
- **Model Checkpoint** :
  - Sauvegarde le meilleur modèle (en termes de perte de validation) pendant l'entraînement.
- **ReduceLROnPlateau** (optionnel) :
  - Réduit le taux d'apprentissage si la perte de validation stagne pendant plusieurs époques.
 

## Alternatives possibles

### **Fonction de perte**
- **Focal Loss** :
  - Utiliser la Focal Loss si les classes sont déséquilibrées pour donner plus d'importance aux exemples difficiles à classer.
- **Pondération des classes** :
  - Utiliser des poids de classe pour donner plus d'importance à la classe minoritaire.

### **Optimiseur**
- **Autres optimiseurs** :
  - Essayer d'autres optimiseurs comme RMSprop ou SGD avec momentum.
- **Taux d'apprentissage adaptatif** :
  - Utiliser un scheduler pour ajuster dynamiquement le taux d'apprentissage pendant l'entraînement.

### 4. **Callbacks**
- **TensorBoard** :
  - Utiliser TensorBoard pour visualiser les métriques d'entraînement en temps réel.
- **Learning Rate Scheduler** :
  - Ajouter un callback pour ajuster manuellement le taux d'apprentissage en fonction du nombre d'époques.

# 6. **Entraînement du modèle**
- **Utilisation du GPU** : Le modèle est entraîné sur un GPU pour accélérer les calculs.
- **Nombre d'époques** : 50 époques maximum, avec un arrêt anticipé si nécessaire.
- **Steps per epoch** : Calculé en fonction de la taille du dataset et du batch size.

# 7. **Évaluation du modèle**
- **Précision sur le test set** : La précision est calculée sur l'ensemble de test pour évaluer les performances du modèle.
- **F1 Score** : Le F1 Score est utilisé pour évaluer la performance du modèle en tenant compte à la fois de la précision et du rappel.
