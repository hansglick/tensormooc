# Strategy Train,Validation,Test

### Méthode .fit()

La méthode `.fit()` permet d'entraîner le modèle sur une partie seulement des données (training set) et d'évaluer celui-ci sur les données de validation. Pour ce faire deux arguments possibles : 
 * `validation_split = 0.2` signifie que 20% du dataset est épargné pour évaluer le modèle
 * `validation_data =` un tuple de matrice de données et un array de label



### Scikit-Learn

```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=p1)
model.fit(X_train, y_train, validation_split=p2)
```


# Régularisation

La régularisation se définit lors de l'instanciation du modele (Sequential()). 3 grandes façons de régulariser :
 * Pénaliser le modèle (norme L1, L2, L1&L2), via les arguments via les arguments `kernel_reguralizer` et `bias_reguralizer` d'un layer
 * Ajouter un layer Dropout
 * Ajouter un layer BatchNormalization
 * Utiliser un callback EarlyStopping



### Pénalisation

 * L2 : `tf.keras.regularizers.l2(0.001)`, pénalisation classique
 * L1 : `tf.keras.regularizers.l1(0.001)` pénalisation à la Tibshirani (pousse les coefficients non pertinents à 0)
 * L1 & L2 : `tf.keras.regularizers.l1_l2(l1=0.05,l2=0.001)`


### Dropout

Le dropout ne s'effectue évidemment que lors du training. Le dropout ne s'effectue pas lors de l'inférence.
Dropout signifie que dans x% un neurone de la couche précédente est assigné à 0
 * `tf.keras.layers.Dropout`
 * `Dropout(rate = 0.5)`


### BatchNormalization

4 parameters. 2 étant des hyper parametres. 2 étant des poids à entraînés :
 * `momentum = .99`
 * `epsiolon = 0.01`
 * `beta` et `gamma` des vecteurs dont les poids sont appris pdt l'entraînement. ils peuvent être initialisés

```
model.add(tf.keras.layers.BatchNormalization(
    momentum=0.95,
    epsilon=0.005,
    axis = -1,
    beta_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
    gamma_initializer=tf.keras.initializers.Constant(value=0.9)
))
```



# Callbacks


### Introduction

Un callback est un objet appelé le plus souvent par les méthodes .fit(), .evaluate() et .predict(). Cet objet permet le calcul et la modification
de variables utilisées lors de l'entraînement, l'évaluation et la prédiction.
Il existe une dizaine de builtin callback qui font des actions specifiques
[documentation](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback?hl=fr)
Tout les objets builtin callback héritent de la classe callback. A chaque fois, on peut utiliser plusieurs callbacks, donc généralement, une liste
d'objets callback est attendue


### Variables accessibles

Au sein de la classe callback, on peut accéder à une ribambelle de variables intéressantes selon les différentes méthodes
 * `logs` : dictionnaire qui store la valeur des métriques, de la loss. Par exemple : logs["loss"] ou logs["mae"]
 * `batch` : le numéro du batch
 * `epoch` : le numéro de l'époque
 * Mais également toutes les variables de l'objet modèle comme :
   * l'objet optimizer : `self.model.optimizer`
   * la valeur de lr en cours : `tf.keras.backend.get_value(self.model.optimizer.lr)`


### Modification de variables

On peut modifier certaines variables du modèle, lors de l'apprentissage, par exemple, pour modifier le learning rate en cours de training
 * `tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_rate)`


### Callback EarlyStopping

Il permet de stopper l'apprentissage lorsque la métrique définie sur le test set commence à se dégrader. Il prend les paramètres suivants :
 * `monitor` : quelle métrique à surveiller, par exemple `val_loss`, `val_accuracy`
 * `patience` : au bout de combien d'épochs sans amélioration, décide t on qu'on arrête l'apprentissage
 * `min_delta` : le seuil au dessus duquel on affirme qu'il y a dégradation
 * `mode` : 'max' ou 'min', le sens du seuil, si c'est max, alors cela signifie qu'on veut la métrique définie maximum (comme l'accuracy par ex)


### Callback Learning Rate Scheduler

 * `tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)`
 * `schedule` : fonction qui prend en input le numéro de l'époque (integer) et le learning rate actuel et renvoie le nouveau learning rate. Egalement possible de faire appel au fonctions lambda pour définir schedule comme dans l'exemple suivant : `tf.keras.callbacks.LearningRateScheduler(lambda x:1/(3+5*x), verbose=1)`



### Callback CSVLogger

 * `tf.keras.callbacks.CSVLogger("results.csv")`
 * Permet de record toutes les informations lors du training, une ligne est égale à une époque, les informations sont la loss et les métriques définies dans la méthode .compile()
   * filename :  le path où enregistrer le fichier
   * separator : le séparateur
   * append : permet d'append un fichier déjà existant, intéressant dans le cadre d'un training en plusieurs étapes
   * exemple : `pd.read_csv("results.csv", index_col='epoch')`


### Callback Lambda

```
tf.keras.callbacks.LambdaCallback(
        on_epoch_begin=None, on_epoch_end=None, 
        on_batch_begin=None, on_batch_end=None, 
        on_train_begin=None, on_train_end=None)
```

Permet de lancer des opérations avec des functions lambda à différents moments du training. En **gras**, les arguments obligatoires

 * `on_epoch_begin` and `on_epoch_end` : **epoch** and **logs**
 * `on_batch_begin` and `on_batch_end` : **batch** and **logs**
 * `on_train_begin` and `on_train_end` : **logs**

Par exemple :

```
epoch_callback = tf.keras.callbacks.LambdaCallback(
    on_epoch_begin=lambda epoch,logs: print('Starting Epoch {}!'.format(epoch+1)))
```


### Callback ReduceLROnPlateau

 * `tf.keras.callbacks.ReduceLROnPlateau`
 * Le callback permet de réduire le learning rate lorsque la loss commence à former un plateau
   * `monitor='val_loss'`: définie la métrique utilisée
   * `factor=0.1`: le facteur par lequel le learning rate doit décroitre. new_lr = factor (x) old_lr
   * `patience=10`: ce qui définit le plateau 
   * `verbose=0`: affichage des informations
   * `mode='auto'`: dans quel sens la métrique est optimisée, "max" quand on veut cette métrique maximum comme l'accuracy. "min" quand on veut cette métrique minimum comme la loss. On peut aussi mettre "auto", à ce moment-là le programme infère la décision lui même en fonction de la métrique définie
   * `min_delta=0.0001`: le changement minimum qui est vu comme une amélioration
   * `cooldown=0`: nombre d'époques à attendre après la modification du learning rate et 'avant que le callback ne revienne à la normal' (rien compris)
   * `min_lr=0`: learning rate minimum si jamais le callback produit un learning rate trop faible

```
tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.1, 
            patience=10, 
            verbose=0, 
            mode='auto', 
            min_delta=0.0001, 
            cooldown=0, 
            min_lr=0)
```


# Autres Tricks

### Class vector to binary OHE matrix
Conversion en matrice OHE d'un vecteur de classe :`tf.keras.utils.to_categorical(random_numpy_aray)`


### Programmation Orienté Objet

Tu peux assigner une fonction à un attribut, assez stylé. `new_lr` est ici le nom d'une fonction. Le keyword super() permet d'hériter des méthodes
et attribuer de la classe parente
```
def __init__(self, new_lr):
   super(LRScheduler, self).__init__()
   #super().__init__() apparemment en python 3 cette syntaxte suffit
   # Add the new learning rate function to our callback
   self.new_lr = new_lr
```



### Data et stratégie de split pour Scikit-Learn

```
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

diabetes_dataset = load_diabetes()
data = diabetes_dataset['data']
targets = diabetes_dataset['target']
train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.1)
```
