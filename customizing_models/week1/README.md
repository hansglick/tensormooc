# L'API fonctionnelle, basic concepts

L'utilisation de l'api fonctionnelle nécessite des layers de type Input (`tf.keras.layers.Input`). L'api fonctionnelle est appelée par le module Model dans `tf.keras.models.Model`. Au lieu d'utiliser des layers comme des objets comme dans l'API séquentielle, on les utilise comme des fonctions. Ces fonctions prennent en paramètre les inputs précédents. Par exemple :

```python
inputs = Input(shape=(32,1))
h = Conv1D(16,5,"relu")(inputs)
h =AveragePooling1D(3)(h)
outputs = Dense(20,"sigmoid")(h)
model = Model(inputs=inputs,outputs=outputs)
```


### Multiple Inputs et Outputs


La puissance de l'API fonctionnelle c'est sa souplesse. Cette souplesse permet l'utilisation de plusieurs couches d'inputs et d'outputs. Dans ces cas là, plusieurs choses vont changer dans les arguments des méthodes et fonctions suivantes :

### tf.keras.models.Model()

 * `inputs` prend en entrée une liste qui représente le nom des objets layers Inputs, par exemple `inputs = [main_inputs,second_inputs]`
 * `outputs` selon la même logique prend entrée une liste, par exemple `outputs = [main_outputs,second_outputs]`


### model.compile()

 * `loss` argument prend désormais OU BIEN :
   * une liste de loss functions objects, par exemple, `loss = ["binary_crossentropy","mse"]`
   * un dictionnaire qui map le nom des layers outputs (non le nom des objets layers outputs), par exemple `loss = {"main_outputs":"binary_crossentropy", "second_outputs":"mse"}`
 * `loss_weights` argument qui représente la part d'importance des deux loss functions dans le calcul de la loss totale, selon la même logique plus haut peut prendre des données de deux formats possibles :
   * une liste de float, par exemple `loss_weights = [1,0.4]`
   * un dictionnaire qui map le nom des layers outputs vers la part d'importance, par exemple, `loss_weights = {"main_outputs":1,"second_outputs":0.4}`
 * `metrics` toujours selon la même logique prend en entrée une liste de listes ou bien un dictionnaire de listes

### model.fit()

 * `X` et `y`, selon la même logique prend ou bien une liste ou bien un dictionnaire par exemple :
   * `X = [Main_X_train,Second_X_train]`
   * `y = [Main_y_train,Second_y_train]`
   * `X = {"main_inputs":Main_X_train,"second_inputs":Second_X_train}`
   * `y = {"main_outputs":Main_y_train,"second_outputs":"Second_y_train"}`


# Variables de Tensorflow

Les variables Tensorflow sont présentes partout dans TensorFlow, les poids des networks sont de ce format et même le learning rate est une tensorflow variable.

### model.weights()

C'est une liste de taille 2 qui représentent le tensor des features et le tensor des bias. Chaque item possède 3 informations
 * le *type de tensor* : kernel ou bien bias
 * la *shape du tensor*
 * le *type* sous lequel sont enregistrés *les poids*
 * le *tensor*


### Créer une variable tensorflow

Les variables/tensor de Tensorflow ressemble beaucoup à celles de numpy.

```python
import tensorflow as tf
myvar = tf.Variable([1,2], dtype = tf.float, name="myvar")
myvar.assign([3.5,-1]) # Ca change la valeur de myvar
x = myvar.numpy() # Le tenseur au format numpy
```

### Différents types

```python
strings = tf.Variable(["hello"],tf.string)
floats = tf.Variable([3.14159,2.71928],tf.float64)
ints = tf.Variable([1,2,3],tf.int32)
complexs = tf.Variable([259 - 7.39j, 1.23 - 4.91j], tf.complex128)
```


### Initialisation à zéro (et pointeurs)

```python
v = tf.Variable(0.0)
w = v + 1 # w dépend de la valeur de v
```

### Basic operations

```python
v.assign_add(1) # addition
v.assign_sub(1) # soustraction
```


# Tensors de TensorFlow

### Création d'un tenseur

```python
x = tf.constant([[1,2,3],[4,5,6]],dtype=tf.float64)
```

### Shaping d'un array

```python
x = np.arange(16)
shape_1 = [8,2]
shape_2 = [4,4]
shape_3 = [2,2,2,2]
tf.constant(x,shape_1)
tf.constant(x,shape_2)
tf.constant(x,shape_3)
```

### Rank et reshape d'un tensor

```python
tf.rank(mytensor)
tf.reshape(mytensor,[8,10])
```

### Ones, Zeros, Identity, Eye

```python
tf.ones(shape=(2,3))
tf.zeros(shape=(2,4))
tf.eye(3)
tf.constant(7,shape=[2,2])
```


### Concatenate tensors

```python
#Concatenate tensors in specific directions
t1 = tf.ones(shape=(2,2))
t2 = tf.zeros(shape=(2,2))
concat1 = tf.concatenate([t1,t2],0)
concat2 = tf.concatenate([t1,t2],1)

```

### Expanding and Squeezing tensors

```python
t = tf.constant(np.arange(24),shape = (3,2,4))
t1 = tf.expand_dims(t,0) # (1,3,2,4)
t2 = tf.expand_dims(t,1) # (3,1,2,4)
t3 = tf.expand_dims(t,3) # (3,2,4,1)

tf.squeeze(t1,0)
tf.squeeze(t2,0)
tf.squeeze(t3,0)
```

### Tensors operations

```python
# Slicing
x = tf.constant([1,2,3,4,5,6,7])
print(x[1:-3]) # 2 3 4

# Matrix multiplication
tf.matmul(c,d)

# Element wise operations
c * d
c + d
c - d
c / c

# absolute value
tf.abs(t)

# Power of tensor
tf.pow(a,a)

# Others
tf.square(a)
tf.exp(a)
tf.cos(a)
```

### Distributions Probablistic

```python
# Normal distribution
tf.random.normal(shape = (2,2), mean = 0, stddev = 1.0)

# Uniform dis
tf.random.uniform(shape = (2,1), minal = 0, maxval = 10, dtype = "int32")

# Poisson dis
tf.random.poisson(shape = (2,1), lam = 3)
```


# Objets d'un NN

 * `model.layers` la liste des layers
 * `model.layers[1].weights`  une variable tensorflow
 * `model.layers[1].get_weights()`  une liste de numpy arrays représentant la paire de tenseurs du layer
 * `model.layers[1].kernel`  la variable tensorflow qui représent les poids classique de la couche
 * `model.layers[1].bias` la variable tensorflow qui représente les biais de la couche
 * `model.get_layer("nom_du_layer").bias`  la variable tensorflow qui représente les biais de la couche 'nom_du_layer'
 * `model.get_layer("mylayer").input` renvoie le tensor d'entrée du layer
 * `model.get_layer("mylayer").output` renvoie le tensor de sortie du layer


### Manipulation de layers et de NN

```python
# Manipulation
model2 = Model(inputs = model.input,outputs = model.get_layer("flatten_layer").output)

# Transfer Learning avec API sequential
model3 = Sequential([
model2,
Dense(10,"softmax","new_top_layer")
])

# Transfer Learning avec API fonctionnelle
new_outputs = Dense(10,"softmax","new_top_layer")(model2.output)
model3 = Model(inputs = model2.inputs,outputs = new_outputs)
```





# Retrouver les features créées par un modèle

```python
# Libs
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model


# Chargement du model
vgg_model = VGG19()

# On prend l'input du modèle
vgg_input = vgg_model.input

# On prend le résultat des activations de toutes les couches
vgg_layers = vgg_model.layers
layer_outputs = [layer.output for layer in vgg_layers]

# On crée un modèle vgg19 exactement le même mais on demande à avoir en output toutes les activations de tout les layers
Extract_Features_Model = Model(inputs = vgg_input, outputs = layer_outputs)

# On le teste sur une random image générée
random_image = no.random.random((1,224,224,3)).astype("float32")
random_image_features = Extract_Features_Model(random_image)
```


# Les inputs et outputs d'un layer

Lorsqu'un layer peut n'avoir qu'un seul input et qu'un seul output, y'a pas de problème :
 * `mylayer.input_shape` / `mylayer.output_shape` : renvoie la shape de l'input/output de mylayer
 * `mylayer.input.name` / `mylayer.output.name` : renvoie le nom de l'input et output du layer


Lorsqu'un layer peut avoir plusieurs inputs et outputs, il faut dans ce cas préciser l'index de l'input / output :
 * `mylayer.get_input_at(index).name` / `mylayer.get_output_at(index).name` : respectivement
 * `mylayer.get_output_shape_at(index)` / `mylayer.get_output_shape_at(index)` : respectivement


# Freezing the model

### trainable argument in layer object

Lors de la définition des layers, on peut utiliser l'argument trainable à False, : `h = Conv2D (...,trainable = False)`

### trainable attribut of layer

Une fois le modèle défini et avant la compilation de celui-ci, on peut accéder aux attributs des layers qu'on veut freezer

```python
model  = load_model('mysavedmodel')
model.get_layer("thislayer").trainable = False
kept_output = model.get_layer("flatten_layer").output
new_head = Dense(5,"softmax","new_head")(kept_output)
new_model = Model(inputs=model.input,outputs=new_head)
```

### attribut trainable du model

On peut également de freezer tout les poids de tout les layers d'un modèle en faisant : `model.trainable = False`


# Device placement

### Accès aux devices installés sur la machine

```python
tf.config.list_physical_devices() # lister les devices de la machine
tf.config.list_physical_devices("GPU") # lister les devices GPU
tf.config.list_physical_devices("CPU") # lister les devices CPU
tf.test.gpu_device_name() # le nom du gpu device
```

### Chaque tensor associé à un device

```pyhton
x = tf.random.uniform([3,3])
x.device # le device assoié à x, GPU:5 veut dire le 5e device GPU

x.device.endswith("CPU:0") # Booléen pour vérifier si un tenseur est associé à un device particulier
```

### Forcer l'association

D'ordinaire, tensorflow alloue les tenseurs aux device de manière automatique, cependant, on peut forcer les assignements

```python
with tf.device("GPU:0"):
x = x = tf.random.uniform([3,3])
someoperations ...
```


# Tricks

### Plot le graph du modèle

`tf.keras.utils.plot_model(model,"mygraph.png",show_shapes=True)`


### Concatenate différentes features

`tf.keras.layers.concatenate([first_layer_object,second_layer_object])`


### Slicing

`x[3:-4] #les 4 éléments à partir de l'item n°3`


### .fit_generator()




### Visualiser une image

```python
import Ipython.display as display
from PIL import image
display.display(Image.open("myphoto.jpg"))
```

### Preprecess une image pour un pretrained model

```python
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image

img_path = "somepath.jpg"
img = image.load_img(img_path,target_size=(224,224)) # resize pour que ca aille dans le modele
x = image.img_to_array(img)
x = np.expand_dims(x,axis = 0)
x = preprocess_input(x)

```


### Visualiser les features extraites d'un CNN

```python
# Extraction des features de x
random_photo_features = Extract_Model(x)


# Extraction des features de layer n° featureid
featureid = 0
myfeature = random_photo_features[featureid]

# my feature est un tenseur pour lequel le premier axe est le numero de l'observation
# Le second axe est l'id du filtre
# Le 3e axe est le channel
# imgs est un tenseur de rang 3, l'input du model
imgs = myfeature[0,:,:]

# Pour le layer 1, on a 64 channels, on peut afficher les 16 premieres
imgs = myfeature[1,:,:]
plt.figure.(figsize=(15,15))
for n in range(16):
ax = plt.subplot(1,3,n+1)
plt.imshow(imgs[:,:,n])
plt.axis("off")
plt.subplots_adjust(wspace=0.01,hspace=0.01)

```


 
### Accéder aux variables trainable et non trainable du model

```python
model.trainable_variables
model.non_trainable_variables
```




