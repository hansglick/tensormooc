# week 4


# Subclassing API

### Model subclassing

```python
# Donc il s'agit de créer un nouveau modèle de class de model functionnel API
class NewModel(Model): # ligne obligatoire pour l'ohéritage
def __init__(self): # ligne obligatoire
super(NewModel,self).__init__() # ligne obligatoire pour l'héritage
# Ici on déclare les layers du modèle
self.dense_1 = Dense(64,"relu")
self.dense_2 = Dense(10)
self.dense_3 = Dense(5)
self.softmax = Softmax()

# Méthode call j'en sais pas plus mais apparamment, c'est la qu'on construit l'architecture du model
# Connection résiduelle
def call(self,inputs):
x = self.dense_1(inputs)
y1 =self.dense_2(x)
y2 = self.dense_3(y1)
concat = concatenate([x,y2])
concat = tf.nn.sigmoid(concat) # tf.nn est un module qui contient les opérations basiques possibles sur les NN
model = self.softmax(concat)
return model
```

Apparemment, y'a des attributs interdits car appartenant déjà à la classe Model. On peut utilsier toutes les opérations du module tf.nn dans le sublassing. Donc apparemment, model et layers seraient assez similaires :
 * **model** : tu peux call un model sur un input, et il te retournera l'output
 * **layer** : tu peux call un layer sur un input et obtenir l'output du layer, c'est ce qu'on fait de l'api fonctionnelle
 * **difference entre les deux** : alors en fait la class model hérite de la classe layer mais elle bénéficie de méthodes en plus comme:
   * .fit()
   * layers property qui storent tout les layers
   * model saving api


### Layer subclassing

```python

# Exemple 1 du mooc
from tensorflow.keras.layers import Layer
class LinearMap(Layer):
def __init__(self,input_dim,units):
super(LinearMap,self).__init__()
w_init = tf.random_normal_initializer() # initialisation des poids du layer
self.w = tf.Variable(initial_value=w_init(shape=(input_dim,units))) # création des poids au bon format

# apparemment tu faire les choses plus simplement en faisant self.add_weight comme thibault neveut, ca définit uene matrice de poids car la class layer possède une méthode add_weight qui permet de créer les poids d'un layer. Y'a deux argument principaux, la shape et l'initializer comme dans
# Les poids sont par défaut trainable, utile pour freezer les poids
self.w = self.add_weight(shape = (input_dim,units),initializer="random_normal")


def call(self,inputs):
return tf.matmul(inputs,self.w) # Donc apparemment là on calcule l'activation du layer, en gros l'output
# du coup produit matricielle entre l'input et les poids du layer


linear_layer = LinearMap(3,2) # là tu crées ton objet layer
inputs = tf.ones((1,3)) # là tu crées ton input
linear_layer(inputs) # tappelles ton layer sur l'input
linear_layer.weights # permet d'afficher les poids du layer qui se trouvent être l'attribut w car ce serait une variable tensorflow




# Exemple 2 de Thibault
class MlpLayer(tf.keras.layers.Layer):

    def __init__(self, units, activations, **kwargs):
        super(MlpLayer, self).__init__(**kwargs)
        # Prend en paramètres la liste du nombre d'units et la liste des activations de chaque couche
        self.units = units
        self.activations_list = activations
        self.weights_list = []


    # Apparemment, build est appelé lorsqu'on connait l'input shape
    def build(self, input_shape):
    # build est utile lorsqu'on veut retarder l'initialisation des poids, par exemple quand on ne connait pas la shape de l'input
    # build est executé la premiere fois qu'on call le layer object, ce qui signifie que les poids ne sont crées que lorsque qu'on appelle l'objet sur un input

        # Donc là on va itérer sur la liste units afin de créer les poids de chaque layer

        i = 0
        for units in self.units:
        # On crée la matrice des poids ici
        # C'est une matrice de dimension input shape, units
        # Elle est stockée dans la liste weights_list
        #
            weights = self.add_weight(
                        name='weights-%s' % i,
                        shape=(input_shape[1], units),
                        initializer='uniform',
                        trainable=True
            )
            i += 1
            self.weights_list.append(weights)
            input_shape = (None, units) # là je comprends jamais pkoi c'est none
        super(MlpLayer, self).build(input_shape) # Alors là je comprends pas le sens
       
    def call(self, x):
        output = x
        # L'activation du layer précédent est initialisé à input
        # Donc c'est ici que ce font les calculs
        # On parcourt toutes les matrices de poids pour calculer l'activation du layer
        # il s'agit donc du produit matricielle entre l'acivation du layer précédent et les poids du layer
        # On retourne l'activation finale
        for weights, activation in zip(self.weights_list, self.activations_list):
 
            output = tf.matmul(output, weights)
            if activation == "relu":
                output = tf.nn.relu(output)
            elif activation == "sigmoid":
                output = tf.nn.sigmoid(output)
            elif activation == "softmax":
                output = tf.nn.softmax(output)
       
        return output






model = tf.keras.models.Sequential()
model.add(MlpLayer([4 , 2], ["relu", "softmax"]))
model.predict(np.zeros((5, 10)))
```

Apparemment quand dans ton custom layer, tu utilises des tensorflow variables qui ne sont pas des poids , tu dois préciser que les poids ne sont pas trainable.


### Build method


#### Exemple 1

```python
class MyLayer(Layer):

    def __init__(self, units, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
        self.units = units

# build est utile lorsqu'on veut retarder l'initialisation des poids, par exemple quand on ne connait pas la shape de l'input
# build est executé la premiere fois qu'on call le layer object, ce qui signifie que les poids ne sont crées que lorsque qu'on appelle l'objet sur un input
# C'est plutot interessant surtout lors de la création d'un model où l'on souhaite que les poids par inférence de l'output du layer précédent
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal')
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros')
    def call(self, inputs):
        return tf.matmul(inputs, self.w)+self.b
```

#### Exemple 2

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Softmax

class MyModel(Model):

    def __init__(self, units_1, units_2, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.layer_1 = MyLayer(units_1)
        self.layer_2 = MyLayer(units_2)

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = tf.nn.relu(x)
        x = self.layer_2(x)
        return Softmax()(x)

# dans ce cas là, lespoids ne sont initialisables qu'après l'instanciation d'un objet de classe model
model = Mymodel(units1 = 32, units2 = 10)
_ = model(tf.ones((1,100)))
model.summary()
```


### Automatic differentiation

#### Exemple simple 1

```python
x = tf.constant(2.0) # la variable à tracker dont on cherche la derivée wrt

with tf.GradientTape() as tape: # création d'un contexte
tape.watch(x) # apparemment là il va tt enregistrer les operations sur x
y = x**2
grad = tape.gradient(y,x) # sort la pente de la fonction y au point x = 2
```

#### Exemple simple 2

```python
x = tf.constant([0,1,2,3], dtype = tf.float32) # c'est comme si t'avais des variables a,b,c,d qu 'on appelé le vecteur x
with tf.GradientTape() as tape:
tape.watch(x)
y = tf.reduce_sum(x**2)
z = tf.math.sin(y)
dz_dy = tape.gradient(z,y)
dz_dy, dz_dx = tape.gradient(z,[y,x]) # en gros les dérivés dz/dy,dz/dx

```

#### Custom loops et linear regression

```python
# Implement a gradient descent training loop for the linear regression model
learning_rate = 0.05
steps = 25

for i in range(steps):
   
    with tf.GradientTape() as tape:
        predictions = linear_regression(x_train)
        loss = SquaredError(predictions,y_train)
        gradients = tape.gradient(loss,linear_regression.weights) # parce que les poids sont trainables, pas besoin de le tape
        # donc il regarde comment la loss est dépendante des linear_regression.weights
        linear_regression.m.assign_sub(learning_rate * gradients[0])
        linear_regression.b.assign_sub(learning_rate * gradients[1])
       
    print("Step %d, Loss %f" % (i,loss))

```

#### Optimizer

```python
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
modelmoi = Mymodel()
loss = MeanSquaredError()
optimizer = tensorflow.keras.optimizers.SGD(learning_rate = 0.05, momentum = 0.9)


with tf.GradientTape() as tape:
current_loss = loss(modelmoi(inputs),outputs)
grads = tape.gradient(current_loss,modelmoi.weights ou modelmoi.trainable_variables)

# pour descendre le gradient, on peut également utiliser un objet optimier préalablement instance
optimizer.apply_gradients(zip(grads,modelmoi.trainable_variables)) # en gros faut lui filer des tuples (gradient de la variable X, valeur de X)
```


#### Optimizer, custom loops

[Exemple custom training](https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough#training_loop)

```python
import tensorflow as tf

for epoch in range(10):
for inputs, outputs in training_dataset:
with tf.GradientTape() as tape:
   current_loss = loss(my_model(inputs), outputs)
   grads = tape.gradient(current_loss, my_model.trainable_variables)
optimizer.apply_gradients(zip(grads, my_model.trainable_variables))

```


#### decorator tf.function

Apparemment, j'ai pas réellement bien compris pourquoi mais en ajoutant un décorateur à certains fonctions impliquant des variables tensorflow, typiquement la fonction qui renvoie le gradient, on transformerait la fonction en graph tensorflow, ce qui accélerait le processus lors du calcul.

```python

# le décorator est placé au dessus de la fonction "graph"
# ce serait ce qui renvoie la loss et les gradients
@tf.function
def grad(model,inputs,targets,wd):
with tf.GradientTape() as tape:
loss_value = loss(model,inputs,targets,wd)
gradients = tape.gradient(loss_value,model.trainable_variables)
return loss_value,gradients

```





# Miscellaneous



### Normaliser les données

Il est preferable de normaliser les données avant entraînement, la raison est que la forme de la fonction de coût a une forme qui permet une descente de gradient plus rapide.

```python
from sklearn.processing import StandardScaler
scaler = StandardScaler() # instanciation
normalized_images = scaler.fit_transform(images)

```

### Shuffle le dataset
Il est recommandé de shuffle le dataset avant de les splitter en train, valid et test. On peut utilier `np.random.shuffle` pour ce faire


### Sauvegarder un modele (révision)

```python
! sudo pip install h5py
model.save("nn.h5")
tf.keras.models.load("nn.h5")

```


### Accumulators as metric

Apparemment pour monitorer l'apprentissage on peut utiliser des accumulateurs de métriques comme ci dessous :

```python
# Loss
train_loss = tf.keras.metrics.Mean(name='train_loss')
valid_loss = tf.keras.metrics.Mean(name='valid_loss')
# Accuracy
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')
```
Ca a pour effet de ne pas avoir à moyenner les métrics sur tout les différents batch vus jusqu'ici dans une boucle comme dans l'exemple suivant :

```python
@tf.function
def train_step(image, targets):
    with tf.GradientTape() as tape:
        # Make a prediction on all the batch
        predictions = model(image)
        # Get the error/loss on these predictions
        loss = loss_object(targets, predictions)
    # Compute the gradient which respect to the loss
    gradients = tape.gradient(loss, model.trainable_variables)
    # Change the weights of the model
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # The metrics are accumulate over time. You don't need to average it yourself.
    train_loss(loss)
    train_accuracy(targets, predictions)
```

En revanche, on a besoin de les reset (particulièremnt quand on change d'époques par exemple) comme ci dessous, (fin de la computation d'une époque) :
```python
valid_loss.reset_states()
valid_accuracy.reset_states()
train_accuracy.reset_states()
train_loss.reset_states()
```

### Résumé des bonnes pratiques au niveau de granularité le plus fin

 1. Créer un générateur des données from directory
 1. Faire en sorte qu'il Shuffle les données et Normaliser les données on the fly (voir les augmente)
 3. Créer un tensorflow dataset avec
 4. Créer un nouveau modèle avec la classe sublassing API
 5. Créer une fonction qui prend en paramètres, le train set, le label set, le model, et qui met à jour les poids du modele avec un optimizer et qui track les métriques appropriées
 6. Créer une fonction similaire pour tracker la performance du modèle sur le validation set
 7. Convertir ces deux fonctions en mode graph grâce au décorateur `@tf.function`
 8. Définir la taille du batch dans le tensorflow data set
 9. Définir le nombre d'époques
 10. Créer une double boucle imbriquées pour entraîner le modèle
 11. Dans la boucle rajoutez des callback pour sauver régulièrement le modèle ou bien pour arrêter son entraînement en cas d'overfitting









