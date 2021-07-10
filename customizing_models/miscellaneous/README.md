

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

