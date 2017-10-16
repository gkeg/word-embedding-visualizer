from __future__ import absolute_import, division, print_function
import codecs
import glob
import multiprocessing
import os
import pprint
import re
import nltk
import gensim.models.word2vec as w2v
import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Download NLTK Modules
nltk.download("punkt")
nltk.download("stopwords")

# Prepare Corpus
book_filenames = sorted(glob.glob("./Corpus/*.txt"))

# Combine all of the text files into one string
corpus_raw = u""
for book_filename in book_filenames:
    print("Reading '{0}'...".format(book_filename))
    with codecs.open(book_filename, "r", "utf-8") as book_file:
        corpus_raw += book_file.read()

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
raw_sentences = tokenizer.tokenize(corpus_raw)
#convert into a list of words
#rtemove unnnecessary,, split into words, no hyphens
#list of words
def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words
#sentence where each word is tokenized
sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))

# Building our model
num_features = 250
min_word_count = 3

# More workers, faster we train
num_workers = multiprocessing.cpu_count()

# Context window length.
context_size = 7

# 1e-5 is a downsampling value known to perform well
downsampling = 1e-3

# Seed for the RNG, to make the results reproducible.
seed = 1

hp2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)  

hp2vec.build_vocab(sentences)

# Pass in all of the necessary training variables
hp2vec.train(
    sentences, 
    total_examples = hp2vec.corpus_count,
    epochs = hp2vec.iter
)

if not os.path.exists("trained"):
    os.makedirs("trained")

hp2vec.save(os.path.join("trained", "hp2vec.w2v"))

hp2vec = w2v.Word2Vec.load(os.path.join("trained", "hp2vec.w2v"))

# Compress the words into a 2d Vector Space
tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)

all_word_vectors_matrix = hp2vec.wv.syn0

all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)

# Plot it!
points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[hp2vec.wv.vocab[word].index])
            for word in hp2vec.wv.vocab
        ]
    ],
    columns=["word", "x", "y"]
)

sns.set_context("poster")

graph = points.plot.scatter("x", "y", s=10, figsize=(20, 12))

harry_x = 0
harry_y = 0

for i, point in points.iterrows():
    # Since matplotlib doesn't show labels automatically, we have to 
    # show it on the actual graph
    if point.word == "Harry":
        harry_x = point.x
        harry_y = point.y
    graph.text(point.x + 0.004, point.y + 0.004, point.word, fontsize = 10)

print("The coordinates of Harry is: (x, y) = " + "(" + str(harry_x) + ", " + str(harry_y) + ")")

print('All Done!')

plt.show()


