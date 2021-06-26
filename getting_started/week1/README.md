
# Tensorflow, en bref

 * Crée en 2017
 * Defaut High Level API est keras
 * Ecrit en C++
 * TensorFlow.js afin de run un modèle dans un browser web
 * TensorFlow Lite afin d'implémenter de l'IA sur des appareils connectés


# Tensorflow, basics

 * Importer la lib, : `import tensorflow as tf`
 * Version de tensorflow : `tf.__version__`
 * Loadder un ficher text en tant que matrice via numpy : `np.loadtxt("filename")`
 * Définir l'architecture d'un modèle via l'API séquentielle : 

### tf.keras.models.Sequential
Utile pour définir l'architecture d'un modèle via l'API séquentielle. Prend en paramètre une liste de layers

### tf.keras.layers.Dense
La couche la plus simple d'un réseau de neurones, prend en paramètres en autres : 
 * Le nombre de units
 * Le nom de la couche
 * La fonction d'activation
 * Si la couche contient un biais ou pas

### compile() Method
Méthode d'un modèle qui permet la compilation du modèle. I.E. les paramètres qui vont permettre l'entraînement de celui-ci. Prend comme paramètres entre autre : 
 * Le nom de l'optimizer utilisé
 * Le nom de la loss function utilisée
 * Une liste de Les métriques à print 

### fit() Method
Méthode d'un modèle qui lance l'apprentissage de celui-ci. Les paramètres sont entres autre : 
 * Le tensor training (au format nump apparemment)
 * Un array représentant les labels (au format numpy également). Il ne doit pas être expanded
 * La taille du batch utilisé
 * Le nombre d'époques pour l'entraînement

### Déclaration de variables
 * myvar = `tf.Variable([[2,1],[1,0]],name = "w")`




# Google Colab

 * [Doc officielle](https://colab.research.google.com/notebooks/welcome.ipynb)


### Import de fichiers sur le drive 

 * `from google.colab import drive`
 * `drive.mount("gdrive")`
 * Then authorize access
 * Then copy paste the password key to the empty box
 * Les fichiers du drive sont désormais accessible avec comme suffixe au path `gdrive/My Drive/`

### Bash commands et le "!"

On peut lancer n'importe quelle commande bash depuis les cellules des notebooks en précédant la commande par un `!`

 * `!ls`
 * `!pip install nympy`
 * `!pip install tensorflow==2`


Une fois que tu installes un package, faudrait cliquer sur `Run Time / Restart Run Time`


### A propos des notebooks

 * On peut uploader des notebooks qu'ils soient sur le drive ou bien sur Github
 * On peut convertir des colab notebooks en jupyter notebooks ca se fait dans edit/.ipynb




# Tensorflow, Manual

 * [official documentation]()
 * Sont listés les modules, les classes et les fonctions de l'API
 * Deux modules importants : **tf.keras** et **tf.data**
