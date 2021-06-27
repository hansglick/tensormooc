
# Keras

Apparemment, vaut mieux regarder la doc sur tensorflow.org que sur keras.io, la doc [tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras)


# API Sequential

On donne à manger à la classe tf.keras.models.Sequential() une liste d'objets de classe tf.keras.layers. La classe Dense de tf.keras.layers est une couche de neurone classique. En paramètres, ca prend entre autre la fonction d'activation et le nombre de neurones. On peut également la taille de l'input, particulierement intéressant pour la première couche

 * `my_list_of_layers = [tf.keras.layers.Dense(10),tf.keras.layers.Dense(5,"relu")]`
 * `mymodel = tf.keras.models.Sequential(my_list_of_layers)`
 * La classe `Flatten` est pas mal, elle prend en entrée un tenseur et renvoie le vecteur déplié correspondant. En paramètre, input renseigne la taille du tensor

Il existe une méthode add() de l'objet model qui permet d'ajouter des couches séquentiellement.



# Objet model

 * attribut `.weights` permet d'accéder aux poids du modèle
 * méthode `.summary()` permet d'afficher le résumé du modèle. La raison pour laquelle, il affiche None dans la taille des layers est parce qu'il attend la taille du batch utilisé
 * attribut layers pour accéder aux layers, lequel possède un attribut weights qui contient les poids
 * méthode .compile() qui permet de définir la loss function ainsi que l'optimizer

# Class Layers


### Dense

 * La couche classique
 * Nombre de neurones
 * Activation function
 * Input shape

### Flatten

 * Unroll un tensor
 * Input_shape

### Conv2D

 * Couche de convolution classique pour les CNN
 * Nombre de filtres, la profondeur
 * kernel_size : la taille de la fenetre. si un integer est renseigné, alors ça veut dire implicitement qu'il s'agit d'une fenêtre de la shape d'un carré 3x3
 * padding : "SAME", le même format 
 * input_shape : L'input shape nécessite de préciser le nombre de channels. il s'attend donc à un triplet, par exemple input_shape = (28,28,1)
 * data_format : argument qui précise les termes de la size définie, si on prend last, pour (28,28,3), alors le dernier integer représente le nombre de channels 


### MaxPooling2D

 * Le fameux layer qui réduit les dimensions
 * La taille : un tuple



# Initialisation des poids


### Arguments

Il semblerait que chaque layer possède comme arguments :
 * bias_initializer : afin d'initialiser les poids "bias" du modèle
 * kernel_initializer : afin d'initialiser les poids "classique" du modèle
 * [La doc des initializers](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)

### Format attendu

3 formats possibles : 

 * string
 * class/objet
 * fonction python custom

##### String

 * 'random_uniform'
 * 'he_uniform'
 * 'zeros'

##### Class / Objet

 * tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
 * tf.keras.initializers.Constant(value=0.4)
 * tf.keras.initializers.Orthogonal(gain=1.0, seed=None)


##### Custom function

```
def my_init(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)

model.add(Dense(64, kernel_initializer=my_init))
```


# .Compile() method

Elle permet de définir la loss function à optimiser ainsi que l'optimizer et les différentes métriques que l'on souhaites tracker pour chaque époque, apparemment.

### Loss function
 * En fait les strings font référence à des classes de l'objet [loss](https://www.tensorflow.org/api_docs/python/tf/keras/losses)
 * "binary_crossentropy"
 * "mean_squared_error"
 * "categorical_crossentropy"

### Optimizer
 * En fait les strings font référence à des classes de l'objet [optimzer](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers) 
 * "sgd"
 * "adam"
 * "adadelta"
 * "rmsprop"


### Metrics
 * En fait les strings font référence à des classes de l'objet [metrics](https://www.tensorflow.org/api_docs/python/tf/keras/metrics)
 * Liste
 * ["accuracy","mae",]





# Fit Method

 * Matrice des données de taille (observation,n features)
 * Vecteur des labels de taille (observations,nombre de classes ou bien 1 selon les options)
 * epoch : le nombre de fois où on passe en revue le dataset
 * batch_size : le nombre d'observations au bout duquel on met à jour les poids. par défaut 32
 * Le résultat de l'appel de laméthode .fit() est un objet dit callback
 * En faisant pd.DataFrame(entrainement.history) on a un dataframe pandas qui représente l'historique de l'apprentissage



# Evaluate Method

 * Matrix with features de dimension (n obs, n features)
 * Matrix Labels de dimension (n obs, nombre de classes ou 1 en fonction des options)
 * Renvoie la loss et les métriques sur les données évaluées
 * Verbose = 2 pour ne pas afficher toute les merdes



# Predict Method

 * Matrix représentant les données à prédire de dimension (n obs, n features)
 * Renvoie une matrice de taille (n obs, nombre de classes)





# Quelques calculs matriciels avec Keras

```
import tensorflow.keras.backend as K

y_true = tf.constant([[0.0,1.0],[1.0,0.0],[1.0,0.0],[0.0,1.0]])
y_pred = tf.constant([[0.4,0.6], [0.3,0.7], [0.05,0.95],[0.33,0.67]])
accuracy =K.mean(K.equal(y_true, K.round(y_pred)))
accuracy
```


# Few Tricks


 * pour rajouter une dimension à un tenseur : matrix[...,np.newaxis]