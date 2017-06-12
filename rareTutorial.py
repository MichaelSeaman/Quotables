# Radim Rehurek's tutorial on word2vec
# https://rare-technologies.com/word2vec-tutorial/


# import modules and set up logging
from gensim.models import word2vec
from gensim.models import KeyedVectors
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = word2vec.LineSentence("quote.txt")
model = word2vec.Word2Vec(sentences, min_count=1)
word_vectors = model.wv
del model
word_vectors.save("word_vectors.bin")
len(word_vectors.vocab)
predicted = word_vectors.seeded_vector("bobby")
word_vectors["bob"]

type(word_vectors.similar_by_vector(.1 + np.zeros(100, dtype=np.float32))[0][0])
