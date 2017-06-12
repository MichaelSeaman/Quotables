# First attempt at generating text w/ word2vec

# Imports
from gensim.models import word2vec
from gensim.models import KeyedVectors
import numpy as np
import h5py
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# Training the word2vec model
raw = open("quote.txt").read()
sentences = [  (sentence + '_').split()   for sentence in raw.splitlines()]
word_list = [word for sentence in sentences for word in sentence]


vector_size = 100
word2vec_model = word2vec.Word2Vec( sentences, min_count=1, size=vector_size)
word_vectors = word2vec_model.wv
del word2vec_model
word_vectors.save("word_vectors.bin")

# If loading was desired, do this instead
# word_vectors = word2vec.KeyedVectors.load("word_vectors.bin")

# Summary Stats
n_words = len(word_list)
n_vocab = len(word_vectors.vocab)
print("# of words: ", n_words)
print("Vocab: ", n_vocab)

# Formatting data
# converting input patterns into the form [samples, time steps, features]
seq_length = 20
n_patterns = n_words - seq_length
dataX = []
dataY = []
for i in range(n_patterns):
    seq_in = word_list[i:i+seq_length]
    seq_out = word_list[i+seq_length]
    dataX.append([word_vectors[word] for word in seq_in])
    dataY.append(word_vectors[seq_out])

print("Total Patterns: ", n_patterns)

X = np.reshape(dataX, (n_patterns, seq_length, vector_size))
y = np.reshape(dataY, (n_patterns, vector_size))

# defining the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]) ) )
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation="softmax") )
model.compile(loss='cosine_proximity', optimizer='adam')


filepath="word-weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
tbCallback = TensorBoard(log_dir='./Graph', histogram_freq=1, write_graph=True, write_images=True)
callbacks_list = [checkpoint, tbCallback]

model.fit(X, y, epochs=40, batch_size= 128, callbacks=callbacks_list)
