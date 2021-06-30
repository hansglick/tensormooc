
# Sauvegarder et Loadder (juste) les poids d'un modèle


## Sauvegarde


#### Callback ModelCheckPoint

Pour sauvegarder les poids d'un modèle, on utilise un callback appelé `ModelCheckPoint`. Plusieurs choses à savoir : 
 * `tf.keras.callbacks.ModelCheckPoint`
 * *path* : sous quel fichier l'enregistrer
 * *save_weights_only* : booléen qui renseigne si l'on doit sauver juste les poids du modèle ou bien les poids et l'architecture
 * *save_freq* : à quelle fréquence les poids doivent-ils être sauvegardés. Par exemple, "epoch" signifie que les poids seront sauvés à chaque époque. Si un integer est renseigné, alors les poids sont sauvés tout les x observations que le modèle a vu lors de l'apprentissage
 * *save_best_only* : permet de dire A tensorflow qu'on va sauver les poids du modèle uniquement si celui-ci est meilleur que le modèle sauvé auparavant. Meilleur dans le sens du critère monitor
 * *monitor* : le critère pour décider si le modèle est meilleur ou pas, par exemple 'val_accuracy'
 * *verbose* : 0,1,2 pour afficher ou pas les informations propres au callback
 * *mode* : le critère doit il etre maximal ou minimum
 * **trick** : pas mal de variables (`epoch`, `batch`, etc.) sont accessibles et de ce fait peuvent être utiles lors du nommage du fichier de sauvegarde, par exemple "checkpoints/mymodel.{batch}.{epoch}.{val_loss.4f}"

Il existe plusieurs formats de sauvegarde des poids. Et en fonction de ce format, les fichiers crées sont différents : 
 * **Native Tensorflow** : pour ça rentrer un path sans extension. Ca crée trois fichiers : `checkpoint`, `mymodel.data-0000-of-0001`, `mymodel.index`
 * **Keras HDF5** : pour ça rentrer un path avec l'extension **.h5**. A ce moment-là seulement un seul fichier est crée : `mymodel.h5`

#### Les fichiers crées

 * `checkpoint` : le plus petit des trois fichiers. Stock simplement des meta data de la sauvegarde
 * `mymodel.data-0000-of-0001` : c'est ce fichier qui contient la valeur des poids du modèle sauvegardés
 * `mymodel.index` : Fichier qui indique à TensorFlow où sont stockées les weights. Utile lorsqu'on run TensorFlow dans un système distribué


#### Méthode .save_weights()

On peut également sauver les poids d'un modèle à la fin d'un apprentissage programmé avec la méthode `.save_weights()` de l'objet model. En appelant cette méthode on va écraser les poids déjà initialisés ou bien existants
 * model.save_weights("mymodel")


## Loader les poids

Pour load les poids d'un modèle sauvegardé préalablement, il suffit de faire appel à la méthode `.load_weights()` de l'objet model. Il faut donc au préalable redéfinir l'architecture du modele.
 * `model.load_weights("mymodel")`
 * `model.load_weights("mymodel.h5")`
 * `model.load_weights(tf.train.latest_checkpoint(checkpoint_dir='checkpoints_every_epoch'))` # afin de loader les poids du checkpoint le plus récent dans un directory




# Sauver et loadder le modèle entier (poids et architecure)

## Sauvegarde

#### Callback ModelCheckPoint

Dans ces cas-là, on indique False pour l'argument save_weights_only. A ce moment-là, les fichiers crées sont quelque peu différents. Premièrement un dossier est crée et à l'intérieur se trouve 3 fichiers et un dossier: 
 * `mymodel/assets/` : apparemment ce dossier peut rester vide selon les lcas
 * `mymodel/saved_model.pb` : l'architecture du modèle
 * `mymodel/variables/variables.data-0000-of-0001` : les poids du modèle
 * `mymodel/variables/variables.index` : pareil fichier qui indique où sont stockés les poids, particulièrement intéressant quand on run le programme dans un environnement distribué
 * **ATTENTION** : dans le cas où le format choisi est le HDF5, il n'y a qu'un seul fichier crée, `mymodel.h5`


#### Méthode .save()

Une fois le modèle entraîné, on peut utiliser la méthode .save() comme dans : 
 * model.save("mymodel")
 * model.save("mymodel.h5")

## Loadder le modèle

```
from tensorflow.keras.models import load_model
model = load_model("mymodelfolder")
model = load_model('mymodel.h5')
```



# Sauver seulement l'architecture

On peut ne sauver que l'architecture d'un modèle et le reload de la façon suivante (attention pour les modèles utilisant l'API non séquentielle, `utilisez tf.keras.from_config()` pour loader l'architecture)

```
# Sans sauvegarder la config
config_dict = model.get_config()
model_same_config = tf.keras.Sequential.from_config(config_dict)



# Avec la sauvegarde en JSON
json_string = model.to_json()

with open('config.json', 'w') as f:
    json.dump(json_string, f)

with open('config.json', 'r') as f:
    json_string = json.load(f)

model_same_config = tf.keras.models.model_from_json(json_string)



# Avec la sauvegarde en YAML
yaml_string = model.to_yaml()
...
```





# Load Pretrained Models

### Keras

Ce [site](https://keras.io/api/applications/) contient des modèles pré entrainés par Keras, la plupart sur la tâche de classification d'images.

Les modèles prétrained de Keras se trouvent dans le module tf.keras.applications
(apparemment, les modèles sont téléchargés dans le dossier ~/.keras/models), par exemple
 * weights : permet d'indiquer à Keras qu'on souhaite les poids appris sur le dataset ImageNet
 * include_top : booléen qui permet d'indiquer à Keras qu'on souhaite que les poids de la dernière couche ne soient pas loaddé (si False), très utile
pour le transfer learning, donc dans ce cas là on charge un réseau sans la dernière couche
 *
```
from tensorflow.keras.applications.resnet50 import ResNet50
model = ResNet50(weights='imagenet',in lucd_top = False)
```

Le module tf.keras.applications.resnet dispose de pas mal de fonctions utiles pour préprocesser les observations et décoder les prédictions :
 * tf.keras.preprocessing.image.load_img : permet de loadder une image et de la resizer
 * tf.keras.preprocessing.image.img_to_aray : permet de convertir une image en numpy array
 * tf.keras.applications.resnet50.preprocess_input : permet d'ajuster le format des observations pour le modèle
 * tf.keras.applications.resnet50.decode_predictions : permet de sortir les labels pour le top N des classes les plus probables 




### Tensorflow Hub

C'est une library qui permet d'accéder à des modèles pretrained. Accès [ici](https://www.tfhub.dev)
 * pour l'installer : `!pip install "tensorflow_hub> 0.6.0"`
 * pour l'import : `import tensorflow_hub as hub`
 * En gros, si je comprends tu importes un hub.KerasLayer object qui se comporte comme un layer Keras classique à mettre dans un modèle instancié par l'api Sequential. Le méthode `.build()` permet de builder le modèle à partir de son hub.KerasLayer object:


```
module_url = "https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/classification/4"
model = Sequential([hub.KerasLayer(module_url)])
model.build(input_shape=[None, 160, 160, 3])
```




# Tricks

### Rajouter une dimension à une image
```
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
lemon_img = load_img("data/lemon.jpg", target_size=(160, 160))
x = img_to_array(img)[np.newaxis, ...]
preds = model.predict(x)
top_preds = pd.DataFrame(columns=['prediction'],
                         index=np.arange(5)+1)
sorted_index = np.argsort(-preds[0])
```

### Accès au poids d'un modèle en numpy
 * `model.weights[0].numpy()`

### On peut remove un model

`del model`

### Bash command dans un notebook

Apparemment, tu peux lancer des bash commands depuis une cellule notebook en précédant la commande par un "!"
 * `! ls -lh myfolder`


### Remove un folder en bash
`! rm -r myfolder`

### Afficher une image dans jupyter

```
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 10, figsize=(10, 1))
for i in range(10):
    ax[i].set_axis_off()
    ax[i].imshow(x_train[i])
```


### Les datasets de keras
```
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
```

### Afficher une image via keras

```
from tensorflow.keras.preprocessing.image import load_img
lemon_img = load_img('data/lemon.jpg', target_size=(224, 224))
```