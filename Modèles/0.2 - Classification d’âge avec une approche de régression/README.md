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


# 6. Générateur de données personnalisé

- Fonction custom_data_generator : Ce générateur personnalisé permet de charger les images en lots (batches) et de les prétraiter à la volée.

    - Chargement des images : Les images sont chargées à l'aide de la bibliothèque PIL (Pillow) et redimensionnées à la taille spécifiée (128x128 pixels).

    - Normalisation des pixels : Les valeurs des pixels sont normalisées entre 0 et 1 en divisant par 255.

    - Extraction de l'âge : L'âge est extrait du nom du fichier et normalisé en le divisant par 116 (l'âge maximum dans le dataset).

    - Gestion des erreurs : Une gestion des erreurs est incluse pour ignorer les fichiers corrompus ou problématiques.

    - Flux continu : Le générateur fonctionne en boucle infinie (while True), ce qui permet de l'utiliser pour l'entraînement de modèles sur de grands datasets.

# 7. Augmentation des données avec ImageDataGenerator

 ### Objectif : Ces transformations augmentent la diversité des données d'entraînement, ce qui aide à réduire le surapprentissage et à améliorer la généralisation du modèle.
 
- Instance ImageDataGenerator : Un générateur d'augmentation de données est créé pour appliquer des transformations aléatoires aux images pendant l'entraînement. Cela inclut :

    - Rotation : Jusqu'à 15 degrés.

    - Décalage horizontal et vertical : Jusqu'à 10 % de la largeur ou de la hauteur de l'image.

    - Zoom : Jusqu'à 20 %.

    - Retournement horizontal : Pour simuler des variations dans l'orientation des visages.


## Alternatives possibles

- Autres techniques d'augmentation : On pourrait ajouter d'autres transformations (luminosité, ajout de bruit, ...)

# 8. Architecture du modèle

- Couches de convolution : Le modèle utilise quatre couches de convolution (Conv2D) avec des noyaux de taille 3x3 et une fonction d'activation ReLU. Ces couches capturent les motifs locaux dans les images (comme les bords, les textures, etc.).

- Nombre de filtres : Le nombre de filtres augmente progressivement (32, 64, 128, 256) pour permettre au modèle d'apprendre des caractéristiques plus complexes au fur et à mesure que la profondeur du réseau augmente.

- Régularisation L2 : La première couche de convolution inclut une régularisation L2 pour pénaliser les poids importants et réduire le surapprentissage.

- Normalisation par lots (Batch Normalization) : Appliquée après chaque couche de convolution pour stabiliser et accélérer l'entraînement en normalisant les activations.

- Pooling : Des couches de max-pooling (MaxPooling2D) avec une fenêtre de 2x2 sont utilisées pour réduire la dimension spatiale des caractéristiques tout en conservant les informations les plus importantes.

- Global Average Pooling : Une couche de GlobalAveragePooling2D est utilisée pour réduire les caractéristiques à un vecteur unidimensionnel avant de passer aux couches denses. Cela réduit le nombre de paramètres et évite le surapprentissage.

- Couches denses : Une couche dense de 128 neurones avec activation ReLU est utilisée pour combiner les caractéristiques extraites. Une régularisation L2 est également appliquée ici.

- Dropout : Un taux de dropout de 0.5 est utilisé pour réduire le surapprentissage en désactivant aléatoirement 50 % des neurones pendant l'entraînement.

- Sortie : La couche de sortie est une couche dense avec un seul neurone et une activation linéaire, adaptée à une tâche de régression (prédiction d'âge).

## Alternatives possibles

- Plus de couches de convolution : On pourrait ajouter plus de couches de convolution pour capturer des caractéristiques plus complexes, mais cela augmenterait le risque de surapprentissage et le temps d'entraînement.

- Pooling alternatif : Au lieu de GlobalAveragePooling2D, on pourrait utiliser une couche Flatten pour aplatir les caractéristiques, mais cela augmenterait le nombre de paramètres.

- Régularisation L1 : Au lieu de L2, on pourrait utiliser une régularisation L1 pour pénaliser les poids de manière différente.

- Dropout différent : Le taux de dropout pourrait être autre (par exemple, 0.3 ou 0.4) ce qui changerait les performances du modèle.

# 9. Compilation du modèle

- Optimiseur : L'optimiseur Adam est utilisé avec un taux d'apprentissage de 0.001. Nous l'avons choisi pour sa capacité à ajuster automatiquement le taux d'apprentissage.

- Fonction de perte : La perte est mesurée par l'erreur quadratique moyenne (MSE), qui est couramment utilisée pour les tâches de régression.

- Métrique : L'erreur absolue moyenne (MAE) est utilisée comme métrique pour évaluer les performances du modèle.

## Alternatives possibles

- Autres optimiseurs : On pourrait utiliser d'autres optimiseurs comme RMSprop ou SGD avec momentum.

- Taux d'apprentissage adaptatif : On pourrait utiliser un scheduler pour ajuster dynamiquement le taux d'apprentissage pendant l'entraînement.

- Autres fonctions de perte : Pour des tâches de régression, on pourrait également utiliser l'erreur absolue moyenne (MAE) comme fonction de perte.

- Métriques supplémentaires : On pourrait ajouter des métriques comme le coefficient de détermination (R²) pour évaluer la qualité des prédictions.

# 10. Résumé du modèle

- model.summary() : Affiche un résumé de l'architecture du modèle y compris le nombre de paramètres à chaque couche. Cela permet de vérifier que le modèle est correctement construit et de comprendre sa complexité.

