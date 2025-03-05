# Objectif

Ce notebook se concentre sur la prédiction de l'âge à partir d'images en utilisant TensorFlow. Il inclut le chargement des données, le prétraitement, l'augmentation des données, et la préparation des modèles pour l'entraînement.

# 1. Chargement des données

- Format des fichiers : Les images sont chargées à partir du dataset UTKFace, où le nom de fichier contient l'âge, le genre et d'autres informations. Ce format est exploité pour extraire les labels d'âge directement.

- Fonction load_image : Utilise tf.io.read_file et tf.image.decode_jpeg pour charger les images en RGB. Ce choix permet de garantir que les images sont dans un format compatible avec les modèles de TensorFlow.

## Alternatives possibles

- Utilisation de tf.data.Dataset : Au lieu de charger manuellement les images, on pourrait utiliser tf.data.Dataset pour créer un pipeline de données qui est plus efficace surtout pour les grands datasets.
    

# 2. Prétraitement des images

- Redimensionnement : Les images sont redimensionnées à une taille fixe (128x128 pixels) pour uniformiser les entrées du modèle.

- Normalisation : Les valeurs des pixels sont normalisées entre 0 et 1 en divisant par 255. Cela améliore la convergence du modèle.

- Augmentation des données : Pour l'entraînement, des techniques d'augmentation sont appliquées, comme le retournement horizontal, l'ajustement de la luminosité, du contraste et de la teinte. Cela permet de réduire le surapprentissage et d'améliorer la généralisation du modèle.

## Alternatives possibles

- Taille des images : La taille de 128x128 pixels est un choix. Une taille plus grande (par exemple 224x224) pourrait capturer plus de détails, mais au détriment de la vitesse d'entraînement.

# 3. Tranches d'âge

- Fonction get_age_range : Convertit l'âge prédit en tranches d'âge (par exemple, 10-19 ans). Cela permet de catégoriser les prédictions pour une interprétation plus intuitive.

- Fonction get_tranche_age : Similaire à get_age_range, mais avec une logique légèrement différente. Cela montre une alternative pour gérer les tranches d'âge.

## Alternatives possibles

- Tranches plus fines ou plus larges : Les tranches d'âge pourraient être ajustées en fonction des besoins. Par exemple, des tranches de 5 ans (20-24, 25-29, etc.) pourraient être plus appropriées pour certains cas d'utilisation mais ici ça ferait beaucoup de tranches d'âge


# 4. Préparation des données pour l'entraînement

- Fonctions process_path_train et process_path_val : Ces fonctions gèrent le prétraitement des images pour l'entraînement et la validation. La version d'entraînement inclut l'augmentation des données, tandis que la version de validation ne l'inclut pas.

- Normalisation de l'âge : L'âge est normalisé en le divisant par 116 (l'âge maximum dans le dataset). Cela permet de ramener les valeurs dans une plage plus adaptée pour l'entraînement.

## Alternatives possibles

- Autres méthodes de normalisation : Au lieu de diviser par 116, on pourrait utiliser une normalisation standard (soustraction de la moyenne et division par l'écart-type) ou une normalisation min-max.


# 5. Visualisation des données

- Fonction visualize_sample_images : Affiche un échantillon d'images avec leurs labels d'âge. Cela permet de vérifier visuellement que les données sont correctement chargées et prétraitées.

- Fonction plot_age_distribution : Trace un histogramme de la distribution des âges pour comprendre la répartition des données.

- Fonction print_statistics : Affiche des statistiques descriptives (moyenne, médiane, écart-type) sur les âges.

## Alternatives possibles
    
- Autres visualisations : On pourrait ajouter des visualisations supplémentaires, comme des graphiques en boîte (boxplots) pour les âges
