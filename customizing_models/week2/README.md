# Datasets


### Le module keras.datasets


Ce module `tensorflow.keras.datasets` contient des datasets représentés eux mêmes sous forme de module. Ainsi pour loadder le dataset cifar10. `load_data` est une des méthodes de l'objet importé. Il renvoie un tuple de paires d'objet. Le call de `load_data` produit deux fichiers sur le disque : `~/keras/datasets/cifar-10-batches-py` et `~/keras/datasets/cifar-10-batches-py.tar.gz`. Tout les datasets accessibles sont [ici](https://keras.io/api/datasets/)

```python
from tensorflow.keras.datasets import cifar10
(xtrain,ytrain) , (xtest,ytest) = cifar10.load_data()
```


Les arguments de la méthode load_data sont multipes :
 * `label_mode="fine"` : signifie qu'on dwl les labels les plus fins, `coarse` indiquerait des labels plus grossiers, par exemple animal and tiger (pour les arguments label_mode à coarse et fine respectivement)
 * `skip_top = 50` : pemet d'ignorer les le top 50 mots les plus fréquents (non discriminant)



# Generator

L'intérêt est de n'uploader en mémoire que le batch qui s'apprête à être utilisé pour l'apprentissage. Sinon, il faut que toutes les donnnées tiennent en mémoire ce qui est rarement le cas. Introduction simple :

```python
def text_file_reader(filepath):
with open(filepath,"r") as f:
for row in f:
yield row

text_generator = text_file_reader("data.txt")
next(text_generator) # output the first line
next(text_generator) # output the second line
```


# Datasets generator

Dans le cas où l'on souhaite utiliser un generator de data pour l'apprentissage, on utilise la méthode `.fit_generator()` comme dans l'exemple suivant :

```python

# Méthode 1
model.fit_generator(training_data = datagenerator_for_training,
validation_data = datagenerator_for_validation,
steps_per_epoch=1000,
epochs=10)

# Méthode 2, pas recommandé sauf cas exceptionnel
for _ in range(1000):
xtrain,ytrain = next(datagenerator)
model.train_on_batch(xtrain,ytrain)


# Evaluation
model.evaluate_generator(datagenerator_eval,steps=100)


#  prediction
model.predict_generator(validation_generator,steps = 1)
```

 * `steps_per_epoch=1000` : comme il s'agit d'un generator, keras ne connait pas la taille du dataset, il faut lui indiquer, ici, toutes les 1000 données générées, il doit considérer ça comme une époque
 * le generator doit renvoyer un tuple de données. Le premier item correspond aux observations, le second correspond aux labels.
 * En pratique, on crée un générateur de données sur le training set puis un générateur de données sur le validation set.



# Data augmentation (images)

### Introduction


```python
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
(xtrain,ytrain) , (xtest,ytest) = cifar10.load_data()

image_generator = ImageDataGenerator(rescale=1/255.,
horizontal_flip = True,
height_shift_range = 0.2,
fill_mode = 'nearest',
featurewise_center = True,
preprocessing_function = myfun)

# Le generateur doit d'abord voir les images dans un premier temps s'il est question de créer des statistiques qui dépendent de la totalité du dataset
image_generator.fit(xtrain)

# Afin de créer un tuple training and labels de batch size = 16
# Afin de créer le générateur de data, il faut appeller .flow() methode sur l'objet generator
train_data_generator = image_generator.flow(xtrain,ytrain,batch_size=16, shuffle = False)
val_data_generator = image_generator.flow(xval,yval,batch_size=16, shuffle = False)

# Ne pas oublier de calculer correctement le nombre de steps nécessaire par epoque
# Correspond donc aux nombre de batch pour le training et le validation dataset
train_steps = train_data_generator.n // train_data_generator.batch_size
val_steps = val_data_generator.n // val_data_generator.batch_size

# entrainement
model.fit_generator(train_data_generator,
epochs = 10,
steps_per_epoch = train_steps)


# evaluate the model
model.evaluate_generator(val_data_generator,steps=val_steps)

# predict observations
# steps = 1, donc tu as les prédictions pourle batch entier
predictions = model.predict_generator(val_data_generator,steps = 1)
```

 * `rescale` :
 * horizontal_flip : booléen qui renverse aléatoirement l'image
 * `height_shift_range` : une partie de l'image est cropé et est remplacé par des pixels average voir `fill_mode`
 * `fill_mode` : par quel pixel remplace t on les pixels cropés
 * `featurewise_center` : moyenner les pixels selon les 3 channels
 * `rotation_range` : rotation de 30 degrés max dans les deux sens aléatoirement
 * `width_shift_range` : comme height sauf que c'est horizontal
 * `preprocessing_fucntion` : argument intéressant qui prend le nom d'une custom function qui va modifier une image


### From directory

Evidemment l'intérêt est de créer un générateur qui va générer des images présentes dans un directory et non à partir de data contenue dans la mémore. Pour ce faire :

```python
classes = ["tiger","cat","dog"]
train_data_generator = image_generator.flow_from_directory(train_path,batch_size=64,classes=classes, target_size =(16,16))

```

 * `classes` : array ou liste qui map les valeurs du array labels vers une description
 * `target_size` : redimmensionner les images générées


# TimeseriesGenerator

Pour de la data augmentation et de la génération de données sur des données séquentielles comme de la musique ou du text, on peut utiliser `tf.keras.preprocessing.sequence.TimeseriesGenerator`

```python
# Crée des séquences de Taille 4 parmi la séquence data
TimeseriesGenerator(data,targets,4)

TimeseriesGenerator(data,
targets,
length = 4,
batch_size=2,
stride = 2,
reverse = True,
sampling_rate=2,
start_index=4555,
shuffle = True)
```

 * `length` : taille des sous séquences générées
 * `batch_size` : le nombre de sous séquences générées
 * `stride` : signale le début de la sous séquence à générer d'après si stride = length alors les sous séquences générées ne s'overlapent pas
 * `reverse` : renverse les sous séquences générées
 * `sampling_rate` : ne prend qu'un item sur 2
 * `start_index` : commencer la génération de sous séquence à un endroit précis dans la séquence
 * `shuffle` : permuter les sous séquences générées du batch
 * `data` : la séquence de laquelle sont générées les sous séquences
 * `targets` : l'item juste après le dernier item de la sous séquence générée, les targets à prédire



# tensorflow.data module


### Exemple simpliste

```python
import tensorflow as tf

# création d'un dataset object
# la shape de ce dataset est de 2
dataset = tf.data.Dataset.from_tensor_slices([[1,2],[3,4],[5,6]])


# On peut itérer sur l'objet crée
for item in dataset:
print(item) # il s'agit de tensors

dataset.element_spec
```

### Exemple avec features and target

```python
# On crée ici deux tuples
# Un pour les features
# Un pour la target
dataset = tf.data.Dataset.from_tensor_slices(
tf.random.uniform([256,4],minval = 1, maxval = 10, dtype = tf.int32),
tf.random.normal([256])
)

# Itérer seulement sur les deux premiers élément de l'objet
for item in dataset.take(2):
print(item)

# exemple secondaire
dataset = tf.data.Dataset.from_tensor_slices((xtrain,ytrain))
```

### Exemple avec un data generator

```python
img_datagen = ImageDataGenerator(width_shift_range = 0.2,horizontal_flip = True)
dataset = tf.data.Dataset.from_tensor_slices(img_datagen.flow,
args = [xtrain,ytrain],
output_types = (tf.float23,tf.int32),
output_shapes = ([32,32,32,3],[32,1]))
print(dataset.element_spec)
# Le premier dimension c'est les batchs


# zipping dataset

```

```python
#apparemment on peut zipper les datsets
tf.data.Dataset.zip
```


```python
# pas bien compris

shaekespeare_dataset = tf.data.TextLineDataset(text_files) #list of textfiles

text_files_dataset = tf.data.Dataset.from_tensor_slices(text_files) # liste e paths
interleaved_shk_dataset = text_files_dataset.interleave(tf.data.TextLineDataset, cycle_length = 9)


```


```python
# Difference entre from_tensor_slices et from_tensors est
# que le premier a une shape = au nombre de features
# que le second a une shape = shape du tenseur input
# from_tensor est plus conciliant il peut prendre des tuples de tensors n'ayant pas la mm dimension première

example_tensor = tf.random.uniform([3,2])
dataset1 = tf.data.Dataset.from_tensor_slices(example_tensor)
dataset2 = tf.data.Dataset.from_tensor(example_tensor)

print(dataset1.element_spec)
print(dataset2.element_spec)

# from_tensor_slices peut également prendre en input un dictionnaire par exemple :
dataframe_dict = dict(pandas_dataframe)
pandas_dataset = tf.data.Dataset.from_tensor_slices(dataframe_dict)
next(iter(pandas_dataset)) # afin d'itérer sur le dataset

```


```python
# Création de dataset depuis un fichier
csv_dataset = tf.data.experimental.make_csv_dataset("data/lalala.csv",batch=1,label_name = "Inlfated")

```

```python
(xtrain,ytrain) , (xtest,ytest) = cifar10.load_data()
dataset = tf.data.Dataset.from_tensor_slices((xtrain,ytrain))
dataset = dataset.batch(16,drop_remainder = True) # le dropremainder sert a remove les observations restantes si on applique que des batch de taille 15, a ce moment la en faisait dataset.element_spec, on voit la taille du batch et non None


# training
history = model.fit(dataset) # entraine sur une seule époque

# training sur 10 epoques
dataset = dataset.repeat(10)
history = model.fit(dataset)

# training sur 10 epochs, autremaniere
dataset = dataset.repeat()
history = model.fit(dataset,steps_per_epoch = xtrain.shape[0] // 16, epochs = 10) # faut lui indiquer le nombre de batch par epoch

# On peut également shufffle le dataset
dataset = dataset.shuffle(100) # signifie que 100 lignes sont shuffles desquelles on va extraire un batch

# On eut également prepreocess chaque observation avec
def mycustomfunction(image,label):
return image/255. , label

def mycustomfunction(x):
x["y"] = 0 if (x["y"] == tf.constant([b'no'], dtype = tf.string)) else 1
return x

dataset = dataset.map(mycustomfunction)
# afin de remove une feature
dataset = dataset.map(lambda a : {key:val for key,val in a.items() if key != 'marital'})


# On peut également éviter cetaines observations en les filtrant
# la fonction doit retourner un booléen
def label_filter(image,label):
return tf.squeeze(label) != 9
dataset = dataset.filter(label_filter)


# On peut également filter avec une lambda function
# afin de remove toutes les observations qui contiennent un marital != "divorced"
dataset.filter(lambda a : tf.equal(a["marital"], tf.constant([b'divorced']))[0])
```




Ne pas oublier qu'à la fin on veut comme spec de notre dataset pour un réseau simple, ceci :
```
shape = (50,)
shape = ()
```

```python
# Afin de créer le train and validation set
training_elements = int(dataset_length * 0.7)
train_dataset = dataset.take(training_elements)
validation_dataset = dataset.skip(training_elements)

```

# tensorflow_datasets library

```python
# pour l'installer
! pip install tensorflow == 2.1.0
! pip install tensorflow_datasets

# Lister les datasets
import tensorflow_datasets as tfds
tfds.list_builders()

# importer un dataset
dataset = tfds.load(name = "kmnist", split = None)
trainds = kmnist["train"]
testds = kmnist["test"]
```

# Tricks

### Mapper des valeurs dans une colonne pandas

```python
df["a"] = df["a"].map(lambda a : 0 if a=="N" else 1)
```

### Column to OHE matrix

```python
matrix = pd.get_dummies(df,prefix="mycol",columns=["Mycol"])
```

### numpy array 1D to binary OHE matrix

`tf.keras.utils.to_categorical(myarray,numberofclasses)`


### Audio data with python

```python

from scipy.io.wavfile import read, write

# Lire un fichier audio et le convertir en numpy array
rate,song = read("mysong.wav")
song = np.array(song)
# song est une matrice de taille 20,000,000,2
# Car 2 est pour la stéréo


# Convetir un numpy array en wav
write("myfile.wav",rate,songarray)

```


### convertis un numpy array binary en scalar

```python
tf.squeeze(array) == 9
```

### iteration

Combien de fois un modèle voit un batch lors de l'apprentissage => le nombre d'itérations

### OHE column in pandas

```python
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
some_features = ["age","education"]
for feature in some_features:
df[feature] = tuple(encoder.fit(df[feature]))
```


### suhffling a pandas dataframe
```python
df = df.sample(frac = 1)
```

