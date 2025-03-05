# Objectif

Ce notebook se concentre sur le prétraitement et la préparation des données pour un modèle de prédiction simultanée du genre et de l'âge. Les données sont organisées en ensembles d'entraînement, de validation et de test, et sont prétraitées pour être utilisées efficacement par un modèle de deep learning.

# 1. Organisation des données

- Fonction preprocess_utkface_dataset :

  - Chargement des images : Les images sont chargées à partir du dataset UTKFace, où le nom de fichier contient l'âge, le genre et d'autres informations.

  - Séparation des données : Les données sont divisées en ensembles d'entraînement (60 %), de validation (20 %) et de test (20 %) à l'aide de train_test_split.

  - Création de dossiers structurés : Les images sont copiées dans des dossiers organisés par genre (male, female) et par tranche d'âge (age_X). Cela facilite la gestion des données et leur utilisation avec des outils comme image_dataset_from_directory.

## Alternatives possibles

- Stratification : Pour garantir une répartition équilibrée des classes (genre et tranches d'âge), on pourrait utiliser une stratification lors de la séparation des données. La stratification est une méthode qui garantit que la répartition des classes dans les ensembles d'entraînement, de validation et de test est proportionnelle à leur répartition dans le dataset original. 

# 2. Prétraitement des images

- Normalisation : Les valeurs des pixels sont normalisées entre 0 et 1 en divisant par 255.

- Augmentation des données : Pour l'entraînement, des techniques d'augmentation sont appliquées, comme le retournement horizontal, la rotation, l'ajustement du contraste et l'ajout de bruit. Cela permet de réduire le surapprentissage et d'améliorer la généralisation du modèle.

- Fonction preprocess_utkface : Cette fonction applique le prétraitement et l'augmentation des données en fonction du mode (train, eval, test).

## Alternatives possibles

- Autres techniques d'augmentation : On pourrait ajouter des techniques comme le zoom, le décalage vertical, ou le mélange de données (mixup) pour améliorer la robustesse du modèle.


# 3. Chargement des données avec tf.data

  - Pipeline de données : Les données sont chargées et prétraitées à l'aide de tf.data.Dataset, ce qui permet un chargement efficace et des opérations de prétraitement sur le GPU.

  - Shuffle et batch : Les données d'entraînement sont mélangées (shuffle) et divisées en lots (batch) pour l'entraînement.

## Alternatives possibles

- Utilisation de ImageDataGenerator : Au lieu de tf.data.Dataset, on pourrait utiliser ImageDataGenerator de Keras, bien que cette méthode soit moins flexible et moins performante.

# 4. Normalisation des labels d'âge

- Âge normalisé : Les labels d'âge sont normalisés en les divisant par 116 (l'âge maximum dans le dataset). Cela permet de ramener les valeurs dans une plage adaptée pour l'entraînement.

## Alternatives possibles

- Autres méthodes de normalisation : On pourrait utiliser une normalisation standard (soustraction de la moyenne et division par l'écart-type) ou une normalisation min-max.

# 5. Visualisation des données

- Affichage d'un échantillon d'images : Un échantillon de 10 images est affiché avec leur genre et leur âge pour vérifier visuellement que les données sont correctement chargées et prétraitées.

# 6. Architecture du modèle

- Entrée unique, sorties multiples : Le modèle prend une image en entrée et produit deux sorties :

  - Une sortie pour l'âge (régression).
  - Une sortie pour le genre (classification binaire).
- Couches convolutives partagées : Les couches convolutives sont partagées entre les deux tâches pour extraire des caractéristiques communes à partir des images.
- Branches séparées :
  - Âge : Une couche dense de 256 neurones avec activation ReLU, suivie d'une couche de dropout (30 %) et d'une sortie linéaire pour la régression.
  - Genre : Une couche dense de 256 neurones avec activation ReLU, suivie d'une couche de dropout (30 %) et d'une sortie sigmoïde pour la classification binaire.

### Régularisation

- Régularisation L2 : Appliquée aux couches convolutives et denses pour éviter le surapprentissage.

- Dropout : Utilisé dans les branches d'âge et de genre pour réduire le surapprentissage.

### Fonctions de perte

- Âge : Erreur quadratique moyenne (MSE) pour la régression.

- Genre : Entropie croisée binaire (BinaryCrossentropy) pour la classification.

### Métriques d'évaluation

- Âge : Erreur absolue moyenne (MAE), erreur quadratique moyenne (MSE) et racine de l'erreur quadratique moyenne (RMSE).

- Genre : Précision (Accuracy).

### Optimiseur

- Adam : Utilisé avec un taux d'apprentissage de 1e-4.

### Callbacks

- Early Stopping : Arrête l'entraînement si la perte de validation ne s'améliore pas pendant 5 époques et restaure les meilleurs poids.

- ReduceLROnPlateau : Réduit le taux d'apprentissage si la perte de validation ne s'améliore pas pendant 3 époques.

## Alternatives possibles

- Autres fonctions de perte : Pour le genre, on pourrait utiliser une perte focal loss si les classes sont déséquilibrées.

- Taux d'apprentissage adaptatif : On pourrait utiliser un scheduler pour ajuster dynamiquement le taux d'apprentissage pendant l'entraînement.

- ModelCheckpoint : Sauvegarder les poids du modèle à chaque amélioration de la perte de validation.

- TensorBoard : Utiliser TensorBoard pour visualiser les métriques d'entraînement en temps réel.
