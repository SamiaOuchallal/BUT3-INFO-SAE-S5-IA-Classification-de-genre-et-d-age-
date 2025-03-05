# Projet : Classification d’âge et de genre en utilisant des réseaux de neurones convolutionnels (CNN)

# 📄 Mise en contexte
Ce projet a pour objectif de développer une application capable de prédire le genre et l’âge à partir d’une image de visage. 

### L’objectif de ce projet va être de réaliser 4 modèles différents en utilisant le jeu de données UTKFace, à savoir :  
* Modèle de Classification de Genre avec CNN 
* Modèle de Classification d’âge avec une approche de régression 
* Modèle de Classification simultanée de Genre et d’âge 
* Modèle pré-entraîné avec l’utilisation du transfer learning et comparatifs avec d’autres modèles pré-entraînés

### Pour permettre par la suite de :
* Créer une interface Gradio
* Déployer l'application et tous les modèles sur HuggingFaces

## 👥 Membres de l'équipe
* CABO India
* GIFFARD Axel
* HAMSEK Fayçal
* OUCHALLAL Samia

La classification de l'âge et du genre est un processus permettant de détecter l'âge et le genre (Homme/Femme) d'une personne en se basant sur des caractéristiques de son visage. On considère les caractéristiques d'une personne par ses traits de visage, ses imperfections, sa pilosité, les rides,etc... \
Toutes ces caractéristiques amènes à détecter l'âge ou le genre d'une personne en fonction d'algorithmes de DeepLearning. Cependant, bien que la détection soit possible, **elle n’en est pas moins certifiée véridique tout le temps.** L'estimation de l'âge varie selon **plusieurs facteurs**, tels que les lumières de l'image, les expressions faciales, le maquillage pour rendre la peau plus "jeune", ...

# Pourquoi réaliser ce projet ? 
<details>
<summary><b>Déroulez pour voir l'ensemble des objectifs : 
</b></summary><br/>
  
- **Exploration et préparation des données** \
  *Analyse du dataset UTKFace (distribution des âges, équilibre hommes/femmes)
  *Prétraitement des images (normalisation, redimensionnement)
  *Augmentation de données pour améliorer la robustesse

- **Comprendre et appliquer les techniques propres au DeepLearning**  
Cela implique d'avoir des notions en mathématiques, science des données, et informatiques pour appliquer des algorithmes d'optimisation linéaire (Fonctions d'activation), de savoir et connaître l'ensemble des paramètres et hyperparamètres utilisés, et de savoir optimiser nos modèles en utilisant des techniques (Dropout, BatchNormalization, learning rate, ...).

- **Comparer différentes architectures de CNN** \
Évaluer les avantages et inconvénients de différentes architectures pour ces tâches spécifiques.

- **Analyser les biais potentiels** \
Identifier les biais potentiels dans les prédictions selon l'éclairage, la qualité de l'image, etc. Les modèles doivent être robustes pour performer de manière constante.

- **Développer une interface utilisateur** \
Créer une interface simple permettant de tester les modèles sur de nouvelles images avec Gradio.
Cela permettra au cours de nos études de présenter ce projet et que les utilisateurs puissent tester l'ensemble de nos modèles.

</details>
  
## 🛠️ Langages et outils
- [Python](https://docs.python.org/)
- [Tensorflow](https://www.tensorflow.org/api_docs).
- [Keras](https://keras.io/).
- [Gradio](https://www.gradio.app/docs).
- [HuggingFaces](https://huggingface.co/)
- [Pandas](https://pandas.pydata.org/docs/)
- [Matplotlib](https://matplotlib.org/stable/index.html)
- [Seaborn](https://seaborn.pydata.org/)


## Résumé de nos modèles 

| **Modèle**                                                                                | **Résumé**                                                                                                                                                                        | **Liens**                                                    | **Métriques**                          |
|-------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|-------------------------------------------|
| Modèle 1 (Genre)                                                                                 | Classification du genre de manière binaire                                                                    | https://github.com/SamiaOuchallal/BUT3-INFO-SAE-S5-IA-Classification-de-genre-et-d-age-/tree/main/Mod%C3%A8les/0.1%20-%20Classification%20de%20genre%20avec%20CNN                         | Binary_accuracy, F1_score, AUC                              |
| Modèle 2 (Age)                                                      | Régression de l'âge                                                                                                                                                    | https://github.com/SamiaOuchallal/BUT3-INFO-SAE-S5-IA-Classification-de-genre-et-d-age-/tree/main/Mod%C3%A8les/0.2%20-%20Classification%20d%E2%80%99%C3%A2ge%20avec%20une%20approche%20de%20r%C3%A9gression | MAE,MSE                                 |
| Modèle 3 (Genre + Age)                                                                             | Classification de genre et régression de l'âge en les combinant simultanément                                                                                                                           | https://github.com/SamiaOuchallal/BUT3-INFO-SAE-S5-IA-Classification-de-genre-et-d-age-/tree/main/Mod%C3%A8les/0.3%20-%20Classification%20simultan%C3%A9e%20de%20Genre%20et%20d%E2%80%99%C3%A2ge                     | MAE,MSE, Binary_accuracy                 |
| Modèle 4 (Transfer Learning)                                                                           | Comparatifs de modèles pré-entraînés (EfficientNetB0/1/2, VGG16, MobileNetV2)                                                                                                                           | https://github.com/SamiaOuchallal/BUT3-INFO-SAE-S5-IA-Classification-de-genre-et-d-age-/tree/main/Mod%C3%A8les/0.4%20-%20Mod%C3%A8le%20pr%C3%A9-entra%C3%AEn%C3%A9%20avec%20l%E2%80%99utilisation%20du%20transfer%20learning                     | MAE,MSE, RMSE, Precision, Recall, F1_score, Accuracy                 |

Réalisé avec https://www.tablesgenerator.com/markdown_tables
</details>


# 🖼️ Le dataset UTKFace
   ## UTKFace
   Le dataset est [UTKFace](https://susanqq.github.io/UTKFace/). C’est un dataset composé de 23708 images avec toutes les ethniques, l'ensemble des genres et de l'âge allant de 0 à 116 ans. Ces images peuvent avoir des tons de couleurs différents, 
  et des variations dans l’expression des visages.
  
  ![alt text](images/Personnes.png)  


## Distribution du genre 

![alt text](images/distribGenre.png)

On repère 52.3 % d'hommes et 47.7 % de femmes.
**Cette distribution indique un potentiel biais concernant la classe minoritaire (femmes) et la classe majoritaire (hommes). Les modèles seront susceptibles de se baser sur la classe majoritaire durant l'entraînement, ce qui peut conduire à un surapprentissage (overfitting). Or, ce biais reste néanmoins faible et ne devrait pas poser d'importants problèmes dans les résultats car une différence de 4 % peut être considéré comme quasi-équilibrée.

## Distribution de l'âge 

![alt text](images/DistributionAge.png)

En faisant cette visualisation, nous remarquons qu'il y a un fort déséquilibre entre les différents âges. Par exemple, il y a énormément d’images de personnes qui ont un âge proche de 26 à 40 ans, peu de jeunes et encore moins de personnes âgées autour de 70 ans. En faisant la moyenne, nous en avons trouvé que le taux le plus important en termes d’âge était de 33 ans.
**En conséquence, le modèle pourrait être plus performant pour estimer l’âge des personnes ayant entre 20 et 40 ans que pour estimer l’âge des personnes entre 60 et 116 ans.**

### ✔️ Tâches pour optimiser les modèles : 
* Normaliser le genre et l'âge
* Réaliser de la Data Augmentation
* Tranche d'âge pour le modèle de l'âge
* Ajuster et expérimenter les paramètres et hyperparamètres
* Expérimenter les tailles de Batch

## Préparation des données 
Nous faisons un "split" des données grâce à la méthode train_test_split de la librairie sklearn
`
    x_train, x_test, y_train, y_test = train_test_split(
        df['image'],
        df['gender_encoded']],
        test_size=0.2,
        random_state=42
    )
    # Split supplémentaire pour validation
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.2, 
        random_state=42
    )
` 

| Dataset      	| Données 	|
|--------------	|---------	|
| Entraînement 	| 18966   	|
| Validation   	| 4742    	|

Nous divisons le dataset en train + val avec 80 % pour le train et 20 % pour le val

## Pré-traitement des données

Les images sont redimenssionnées à une taille uniforme : **224x224** pour les modèles pré-entraînés (conventions de ces modèles) et **128x128** pour le reste des modèles.

### Normalisation de l'âge et du genre

* On vient normaliser l'âge en prenant en compte l'âge supposé maximale dans le dataset,
```python
normalized_age = tf.cast(age,tf.float32) / 116.0
```
* On vient normaliser le genre en divisant par 255 pour transformer la valeur des pixels entre 0 et 1. Ces valeurs vont s'adapter plus rapidement lors de l'entraînement
  ```python
  image=image/255.0
  ```

### Data Augmentation
La technique de la Data Augmentation va permettre d'augmenter la taille du dataset : En créant de nouvelles images à partir des images existantes, on multiplie artificiellement la quantité de données d'entraînement disponibles.
Elle va permettre également de réduire le risque d'overfitting en l'empêchant de se baser sur les mêmes visages.
Elle inclue des transformations de type : 
* Rotation
* Flip
* Zoom
* Luminosité/contraste/saturation

```python
image = tf.image.random_crop(image, size=(96, 96, 3))
        image = tf.image.resize(image, size=size)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
```


</br>
Ce qui peut donner pour un set d'images comme celui-ci : 

![image](https://github.com/user-attachments/assets/989e433c-5e9f-4831-84fd-d343160c7999)


## 📄 Hyperparamètres utilisées
* Type d'activation : (relu, Sigmoid,softmax)
* learning rate : Généralement 0.0001
* Taille du batch : Généralement 32
* Optimiseur : Généralement Adam
* Dropout : Testé sur 0.2, 0.3 et 0.5
* Kernel_regularizer : Généralement 0.0005
* kernel_size
* strides
* pooling_size=Généralement MaxPooling

### Modèles pré-entraînés + Graphiques
Chaque image est en dimension 224x224

<details>
  <summary><b>EfficientNet (B0/B1/B2)</b></summary><br/>
  
  ![ image](images/EfficientNet-Architecture-diagram.png)

  # EfficientNetB0 :
  ```python
  base_model_efficientnet = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
  ```
![image](images/effib0Courbes.png)
![image](images/b0ConfuGenre.png)
![image](images/distribeffib0.png)
![image](images/scattereffi0.png)




 ### Résultats (50 epochs)

  # EfficientNetB1 :
  ```python
  base_model_efficientnet = EfficientNetB1(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
  ```
![image](images/courbeffib1.png)
![image](images/confueffib1genre.png)
![image](images/scatteploteffib1.png)


 ### Résultats (50 epochs)

  # EfficientNetB2 :  
  ```python
  base_model_efficientnet = EfficientNetB2(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
  ```

 ### Résultats (50 epochs)
  ![image](images/EfficientNetB2Courbes.png)
  ![image](images/EfficientNetB2DistributionErreursAge.png)
  ![image](images/EfficientNetB2PredAge.png)



</details>
<details>
  <summary><b>VGG16</b></summary><br/>
  
  ![image](images/VGG16.png)
  
   ### Résultats (50 epochs)

  ![ image](images/VGG16Courbes.png)
  ![ image](images/distribVGG16.png)
  ![ image](images/ConfuMatrixVGG16.png
)

  

</details>

<details>
  <summary><b>MobileNetV2 </b></summary><br/>
  
  ![image](images/MobileNetV2-architecture.png)

  


  
  
   ### Résultats (50 epochs)
   
  ![image](images/MobileNetV2Courbe.png)
  ![image](images/distribmobile.png)
  ![image](images/scatterplotmobile.png)
  ![image](images/confugendermobile.png)

</details>

<details>
  <summary><b>ResNet50</b></summary><br/>
  
  ![image](images/archiresnet50.png)

</details>

| Modèles             | Accuracy - Genre  | AUC | F1_Score | Precision | Recall | MAE  | MSE    | RMSE  | Age Accuracy (10 ans d'écart en %) | #Params |
|---------------------|-------------------|-----|----------|-----------|--------|------|--------|-------|------------------------------------|---------|
| Genre               |        86.33 %    |0.935|   0.8728 |    0.8627 |  0.8832|  -   | -      | -     | -                                  | 2,600 M |
|         Age         |         -         |  -  |     -    |     -     |    -   | 5.81 |  63.45 |  9.24 |                                    | 423,361 |
| Genre + Age         |        0.91       |  -  |          |           |        | 8.22 |  96    |  11.33|                                    | 8,779 M |
| TA - EfficientNetB2 |        0.9373     |  -  |   0.94   |    0.95   |  0.93  | 5.52 |  62.79 |  7.92 |               83.64%               | 7,499 M |
| TA - EfficientNetB0 |        0.90       |  -  |   0.90   |    0.89   |  0.91  | 7.58 | 108.06 | 10.40 |               73.45%               | 4,383 M |
| TA - EfficientNetB1 |        0.90       |  -  |   0.91   |    0.91   |  0.91  | 6.86 | 92.93  | 9.64  |               77.01%               | 8,417 M |
|      TA - VGG16     |        0.88       |  -  |   0.89   |    0.89   |  0.89  | 8.02 | 125.05 | 11.18 |               71.19%               | 15,7 M  |
|   TA - MobileNetV2  |        0.93       |  -  |   0.93   |    0.93   |  0.93  | 6.07 |  76.52 |  8.75 |                80.64               |   3.2M  |

En résumé, le meilleur modèle pré-entrainé est EfficientNetB2 qui prime avec 93 % d'accuracy pour le genre, avec 5.52 d'MAE. Autrement dit, le modèle peut se tromper de genre avec une probabilité de 7%, tandis que pour l'âge, le modèle est susceptible de se tromper entre 5 et 6 ans d'écart. Il peut aussi être ammené à une probabilité de se rapprocher de 83.64 % entre 0 et jusqu'à 10 ans de plus. 


## ✨ Demos
- [Hugging Face Spaces Application Demo ](https://huggingface.co/spaces/samsam2908/ia_sae)
- [Présentation Diapo Canva](https://www.canva.com/design/DAGfd7WHdOo/nloRVfCN5UlDWJWTtCqofQ/edit?utm_content=DAGfd7WHdOo&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)
