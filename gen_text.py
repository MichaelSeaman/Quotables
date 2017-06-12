import numpy as np
import h5py
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from gensim.models import word2vec
from gensim.models import KeyedVectors

def main():
    if len(sys.argv) == 1:
        print("Weights file was not provided")
        sys.exit(0)
    elif len(sys.argv) == 2:
        weights_file = sys.argv[1]
        vectors_file = "word_vectors.bin"
    else:
        weights_file = sys.argv[1]
        vectors_file = sys.argv[2]

    print("Weights found at ", weights_file)
    print("Vector mappings found at ", vectors_file)


    # Loading vectors
    word_vectors = KeyedVectors.load(vectors_file)

    # Loading text
    raw = open("quote.txt").read()
    sentences = [  (sentence + '_').split()   for sentence in raw.splitlines()]
    word_list = [word for sentence in sentences for word in sentence]

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
    for i in range(n_patterns):
        seq_in = word_list[i:i+seq_length]
        dataX.append([word_vectors[word] for word in seq_in])

    vector_size = len(word_vectors[word_list[0]])
    X = np.reshape(dataX, (n_patterns, seq_length, vector_size))

    # defining the LSTM model
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])) )
    model.add(Dropout(0.2))
    model.add(Dense(vector_size, activation="softmax"))

    # Loading Weights
    model.load_weights(weights_file)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print(generate_words(100, model, dataX, word_vectors))

def generate_words(n, model, dataX, word_vectors):
    # Generates n words using the model, dataX for seed, and the word_vectors used in generation
    out = []
    # pick a random seed
    start = np.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    print("Seed:")
    print("\"", ' '.join([ word_vectors.similar_by_vector(vec)[0][0]  for vec in pattern   ]), "\"")
    # generate words

    for i in range(n):
        x = np.reshape( pattern, (1, len(dataX[0]), len(dataX[0][0]))  )
        prediction = model.predict(x, verbose=0)[0]
        result = word_vectors.similar_by_vector(prediction)[0][0]
        out.append(result + " ")
        pattern.append(prediction)
        pattern = pattern[1:len(pattern)]
    print("\nDone Generating")
    return "".join(out)

if __name__ == "__main__":
    main()
