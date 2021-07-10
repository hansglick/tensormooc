
# week 3

# Preprocessing sequence data

### padder les séquences

```python
tf.keras.preprocessing.sequence import pad_sequences

test_input = [[1,3],[3,9,11],[0]]
preprocessed_input = pad_sequences(test_input,padding="post",maxlen=5)

```
 * `padding` : `post` / `pre` : prepadd les sequences par rapport a la sequence la plus longue
 * `maxlen` : tronque le séquences de telle sorte à ce qu'elles soit de taille maxlen. Tronque soit au début soit à la fin en fonction du keyword `truncating`
 * `truncating` : `post` / `pre` : indique si la troncature à lieu à gauche (pre) ou à droite (post)
 * `value` :  la valeur pour le padding
 * Fonctionne très bien avec les liste de listes


### Imdb keras dataset

```python
import tensorflow.keras.datasets.imdb as imdb
(xtrain,ytrain), (xtest,ytest) = imdb.load_data(num_words = 1000, skip_top = 10)
xtrain[0] # liste de integers représentant les mots d'une revue
ytrain[0] # 1 / 0 1 si positif review
imdb_word_index = imdb.get_word_index() # le mapper des integers vers les mots réels
index_from = 3 # apparemment c'est la valeur par défaut va comprendre, il faut donc modifier le dictionnaire
imdb_word_index = {key:value + index_from for key,value in imdb_word_index.items()}
inv_imdb_word_index = inverse du dic
imdb_word_index["salut"] # retourne l'integer correspondant

```

### Padding sequences

#### exemple 1

```python
from tensorflow.keras.layers import Masking
masking_layer = Masking(mask_value = 0)
preprocessed_data = preprocessed_data[...,np.new_axis] # tensor avec les dimensions batch size, sequence length, features
masked_input = masking_layer(preprocessed_data)
print(masked_input._keras_mask) # affiche n liste de booléens qui correspondent à True si value original et False si value padding
```


#### exemple 2

```python
padded_xtrain = tf.keras.preprocessing.sequence.pad_sequences(xtrain,maxlen=300,padding="post",truncating="pre")
padded_xtrain.shape # 25000, 300
padded_xtrain = np.expand_dims(padded_xtrain,-1) # -1 ou 0 donne pareil
tf_xtrain = tf.convert_to_tensor(padded_xtrain, dtype="float32") # probablement convertis un numpy array dans un tensor
masking_layer = Masking(mask_value = 0.0)
masked_xtrain = masking_layer(tf_xtrain)
```


### Tokenizing

Petits blocks de scripts pour en savoir plus sur les tokenizers, voir [la doc officielle](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer)


#### Exemple 1

```python
text_string = text_string.replace('—', '') # remplacer des caractères
sentence_strings = text_string.split('.') # splitter une string vers une liste de strings
```


#### Exemple 2

```python
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=None,
                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                      lower=True,
                      split=' ',
                      char_level=False,
                      oov_token='<UNK>',
                      document_count=0)
```
 * `filters` : les caractères qu'on veut remove
 * `split` : caractère utilisé pour split les strings en mots
 * `oov_token` : string utilisé pour des mots hors du vocabulaire
 * `num words` : le nombre maximum de mots à garder dans le vocabulaire. Si None tout les mots sont gardés
 * `lower` : lowerising les mots, défaut à True
 * `char_level` : tout les caractères sont tokenizés si True, défaut False



#### Exmple 3

```python
# une fois l'objet tokenizer instancié, on l'utilise pour tokenizer les data
# Pour builder le tokenizer vocabulary
tokenizer.fit_on_texts(sentence_strings) # sentence_strings est une liste de strings

# On peut accéder aux paramètres du tokenizer
tokenizer_config = tokenizer.get_config()
tokenizer_config.keys()

# Le dictionnaire des fréquences des mots
# le dictionnaire de mapping des words to integers
tokenizer_config['word_counts']
tokenizer_config['word_index']
# les index correspondent au rang fréquentielle des mots du corpus

# Afin de tokenizer les data
sentence_seq = tokenizer.texts_to_sequences(sentence_strings)

# Afin de détokenier les sentences tokenizées
tokenizer.sequences_to_text(sentences_seq)


```




# Embedding Layer

#### Exemple 1

```python
from tensorflow.keras.layers import Embedding
embedding_layer = Embedding(input_dim = 1000,
output_dim = 32,
input_length=64,
mask_zero = True)
# input_length : la taille des sequences requis ou pas en fonction es cas
# interpret les zeros comme des padding values

# inputs de taille batch_size,taille max des sequences
embedded_input = Embedding(inputs) # le tensor en sortie sera de shape taille du batch, length max des sequence, embedding dimension
embedded_input._keras_mask # liste de listes de booléens en fonction des padding values et originals
# Le network va ignorer les padding values,il ne s'en servira pas pour mettre à jour les poids
# Apparemment, dans ces cas là, t'aurais pas besoin de Masking() layer
```

#### Exemple 2

```python
sequence_of_indices = tf.constant([[[0],[5],[15]]])
# la dimensionalité est telle que ça respecte(batch size, sequence length, features)

sequence_of_embeddings = embedding_layer(sequence_of_indices)
sequence_of_embeddings.shape # 1,4,1,16 taille du batch, taille de la sequence, features number, embedding space dimension

# Pour afficher tout les embeddings
embedding_layer.get_weights()[0] # chaque row est l'embedding d'un word

# Pour afficher l'embedding de mot n°14
embedding_layer.get_weights()[0][14]

```

#### Exemple 3

```python
masking_embedding_layer = tf.keras.layers.Embedding(501,16,mask_zero=True)
masked_sequence_of_embeddings = masking_embedding_layer(sequence_of_indices) # les 0 seront vu comme des valeurs de padding
```




# RNN layers


Rappel rapide schématisé sur les [RNN](https://www.researchgate.net/figure/An-example-of-bidirectional-LSTM-tagger-architecture_fig1_330276457). Apparemment par défaut le RNN layer renvoie le final output par tout les hidden state de la séquence. Il prend des sequences de n'importe quelle taille. L'input attend un tenseur de dimensions (batch size, fixed length, n features by timestamps, autrement dit l'embedding space)


#### Simple RNNs

```python
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Bidirectionnal

model = Sequential([
Embedding(1000,32,64) # batch size, 64, 32
SimpleRNN(64,"tanh") # batch size, 64
Dense(5,"softmax") # batch size, 5
])

```

[architecture de rnn bidirectionnel](https://stackoverflow.com/questions/62991082/bidirectional-lstm-merge-mode-explanation)

#### Bidirectionnal Stacked RNNs

```python
return_sequences = True # dans tout les layers de sequences , il est a False par défaut
# il permet de retourner les hidden state de la sequence et non seulement le final output
# du coup le tensor de sortie est de hsape differente, ce qui nous permet de réer des stacks de LSTMs

inputs = Input(shape = (None,10))
h = Masking(mask_value=0)(inputs)
h = Bidirectionnal(LSTM(32,return_sequences=True))(h)
h = Bidrectionnal(LSTM(64))h # Dans ces cas, on concatene les deux outputs finaux et donc on a une shape de batch size, sequence len, 64*2
# apart si on décide de le sommer grace a largument merge_mode = 'concat' par défaut, on a donc une concaténation mais on aurait pu prendre, ca peut prendre aussi les valeurs de  'ave' ou 'sum' ou 'mul'
outputs = Dense(5, activation = "softmax")(h)
model = Model(inputs,outputs)


# un layer bidirectionnel, plus académique
Bidirectionnal(layer = LSTM(8),merge_mode = "sum", backward_layer = GRU(8, go_backwards = True))
```

#### Statefull RNNs

Il arrive que lors de très longues sequences, comme en time series, ou bien en speech audio, il faille splitter les séquences en petits chunks, dans ces cas là, on se retrouve avec des batch de chunks. Il faudrait donc que d'un chunk à l'autre s'il sagit de la mm sequence, le statefull du RNN soit persistent entre les batchs. Pour que les layers soit statefull entre les batchs, il faut renseigner l'argument `stateful = True`. Il faut alors égalment renseigner l'argument `batch_input_shape = (2,None,1)` (uniquement des batch de 2 séquences). Apparemment, il faudrait renseigner cet argument dès le premier layer. J'avoue que j'ai pas bien compris comment on utilisait la méthode reset mais c'est essentiel pour maîtriser les statefull RNNs.


#### Example end-to-end Statefull RNN

```python

# STEP 1 - Création d'un tokenizer
tokenizer_object = Tokenizer(num_words=None,
                             lower= False,
                             char_level = True,
                             filters=None)
tokenizer_object.fit_on_texts(list_of_strings)



# STEP 2 - Encodage des sequences via le tokenizer
encoded_sentences = tokenizer.texts_to_sequences(list_of_strings)



# STEP 3 - Padder les séquences de sorte à pouvoir batcher
padded_encoded_sentences = pad_sequences(sequences = sequence_chunks,
maxlen = 500,
truncating = "pre",
value = 0,
padding = "pre")



# STEP 4 - Préparer la matrice d'inputs et la matrice outputs
# Chaque mot doit prédire le mot d'après, faut donc opérer un décalage d'un mot sur l'autre
# A ce moment la matrice est de dimensions num observation, taille des sequences padded - 1
input_array = padded_encoded_sentences[:,0:-1]
output_array = padded_encoded_sentences[:,1:]



# STEP 5 - Etant donné que toutes les phrases se suivent, et qu'on fait des batch on veut renvoyer le state du RNN du batch b n-1 au batch n
# On veu donc construire les bach de telle sorte à ce que le ieme element du batch b soit le précédent du ieme element du batch b+1
input_seq_statefull = [0,22,45, 1,23,46, ...] # les indices des sentences dans le cas où l'on a des batch de taille 3
output_seq_statefull = # la même chose



# STEP 6 - Création des training et validation set et de leur tensorflow dataset respectif
... # creation des numpy array respectifs train, test
train_data = tf.data.Dataset.from_tensor_slices((train_input_array,train_target_array))
train_data = train_data.batch(32,remainder=True)
valid_data = tf.data.Dataset.from_tensor_slices((output_input_array,output_target_array))
valid_data = valid_data.batch(32,remainder=True)



# STEP 7 - Création du model
layer_embedding = Embedding(input_dim = vocab_size,
output_dim = 256,
mask_zero=True,
batch_input_shape=(batch_size,None))

layer_gru = GRU(units = 1024,
return_sequences = True,
stateful = True)

layer_dense = Dense(units = vocab_size)

model = Sequential([layer_embedding,layer_gru, layer_dense])



# STEP 8 - Training the model
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(filepath='./models/ckpt',
                                                       save_weights_only=True,
                                                       save_best_only=True)
model.compile(optimizer='adam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
metrics=['sparse_categorical_accuracy'])

history = model.fit(train_data, epochs=15, validation_data=valid_data,
                    validation_steps=50, callbacks=[checkpoint_callback])
# Ici on utilise from_logits = True car on n'a pas mis d'activation function
# Si on avait mis un softmax, alors on aurait pas eu à mettre from_logits = True



# STEP 9 - Predictions
# token_sequence est une liste de listes qui est censé représenter un batch de phrases pour lequel on essaie de prédire le mot d'après
# Mais ici on n'est intéressé que par l'état du dernier mot afin de générer la suite
predictions = model.predict(np.array(token_sequence))
predictions = np.expand_dims(predictions[0][-1],axis = 0) # on ne prend que le dernier state, autrement dit l'encodage de la phrase d'entrée
# L'encodage est de taille égale au nombre d'unité du GRU, il s'agit de logits car aucune fonction d'activation ne sert de mapper
# Du coup pour générer un mot il suffit de tirer dans l'array de logit en output, par exemple
next_word_integer = int(tf.random.categorical(logits,num_samples=1))



# STEP 10 -
init_string = 'ROMEO:'
num_generation_steps = 1000
token_sequence = tokenizer.texts_to_sequences([init_string])
initial_state = None
input_sequence = token_sequence

for _ in range(num_generation_steps):
    logits = get_logits(model, input_sequence, initial_state=initial_state)
    sampled_token = sample_token(logits) # tirer au hasard un entier
    token_sequence[0].append(sampled_token)
    input_sequence = [[sampled_token]]
    initial_state = model.layers[1].states[0].numpy()

# decodage de la séquence
tokenizer.sequences_to_text(token_sequence)[0][::2]
```



# Tricks


#### **Global Average Pooling 1d**

prend en entrée un cube, par exemple (2,3,4) et renvoie la moynne sur l'axe 1. L'output est de (2,4)

#### **Accéder au drive dans google colab**
 
```python
from google.colab import drive
drive.mount("/content/drive")
```

#### **pas de taille définie pour un input**

```python
review_sequence = tf.keras.Input((None,))

```

#### **Tensorflow embedding projector**

[Vizualize embeddings in low space](https://projector.tensorflow.org/)


# Questions

 * Revoir les **statefull RNNs**, théorie et en keras


#### **loss from_logit = True**

Les deux options suivantes sont strictement identiques [source](https://datascience.stackexchange.com/questions/73093/what-does-from-logits-true-do-in-sparsecategoricalcrossentropy-loss-function):
 * un layer final sans activation et une loss entropy from_logit = True
 * un layer final avec softmax activation et une loss entropy classique





