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

vec_model = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)  

vec_model.build_vocab(sentences)

# Pass in all of the necessary training variables
vec_model.train(
    sentences, 
    total_examples = vec_model.corpus_count,
    epochs = vec_model.iter
)

if not os.path.exists("trained"):
    os.makedirs("trained")

vec_model.save(os.path.join("trained", "trained_model.w2v"))

vec_model = w2v.Word2Vec.load(os.path.join("trained", "trained_model.w2v"))

# Compress the words into a 2d Vector Space
tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)

all_word_vectors_matrix = vec_model.wv.syn0

all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)

# Plot it!
plt.style.use('ggplot')

points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[vec_model.wv.vocab[word].index])
            for word in vec_model.wv.vocab
        ]
    ],
    columns=["word", "x", "y"]
)

sns.set_context("poster")

graph = points.plot.scatter("x", "y", s=10, figsize=(20, 12))

harry_x = 0
harry_y = 0

# for i, point in points.iterrows():
    # Since matplotlib doesn't show labels automatically, we have to 
    # show it on the actual graph
  #   if point.word == "Harry":
    #     harry_x = point.x
      #   harry_y = point.y
    # graph.text(point.x + 0.004, point.y + 0.004, point.word, fontsize = 10)

print("The coordinates of Harry is: (x, y) = " + "(" + str(harry_x) + ", " + str(harry_y) + ")")

print('All Done!')

def plot_region(x_bounds, y_bounds):
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) & 
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1])
    ]
    
    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)


def nearest_similarity_cosmul(start1, end1, end2):
    similarities = vec_model.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
    )
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2

# Just for better colors
plt.show()


