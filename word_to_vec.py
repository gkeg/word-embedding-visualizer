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
import pickle
import sys
from highlight import label_area

def train_data(save_pickle_points_path):
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
    # Convert into a list of words
    # remove unnnecessary characters, split into words, no hyphens
    def sentence_to_wordlist(raw):
        clean = re.sub("[^a-zA-Z]"," ", raw)
        words = clean.split()
        return words
    # sentence where each word is tokenized
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(sentence_to_wordlist(raw_sentence))

    # Giving a status update
    print("Here's an example of what our wordlist looks like right now")
    print(raw_sentences[100])
    print(sentence_to_wordlist(raw_sentences[100]))


    # Building our model
    num_features = 250
    min_word_count = 3

    # More workers, faster we train
    num_workers = multiprocessing.cpu_count()

    # Context window length.
    context_size = 7

    # 1e-3 is a downsampling value known to perform well
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
    print("We're gonna train the model now...")
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
    print("We're just gonna compress the dimensions... hang tight!")
    # Compress the words into a 2d Vector Space using t-distributed stochastic neighbour embedding
    tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)

    all_word_vectors_matrix = vec_model.wv.syn0

    all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)

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
    # Check if we want to save the points to a pickle
    if (save_pickle_points_path):
        points.to_pickle(save_pickle_points_path)
    return points

def plot_data(points, highlighted_word):
    plt.style.use('ggplot')
    sns.set_context("poster")
    points.plot.scatter("x", "y", s=10, figsize=(20, 12))
    if highlighted_word is not None:
        label_area(points, highlighted_word)
    print("Close the window to continue")
    plt.show()

def main(argv):
    points = None
    highlighted_word = None
    save_pickle_points_path = None

    should_load_points = input("Would you like to load pre-trained points from a file? y/n \n").lower().strip()
    if (should_load_points == 'y'):
        load_points_path = input("Please enter the relative path where you would like to load from: \n").strip()
        while True:
            try:
                points = pd.read_pickle(load_points_path)
                break
            except FileNotFoundError:
                print("Oops! We couldn't find that file, please try again. \n")

    should_save_points = input("Would you like to save the trained points to a path? y/n \n").strip()
    if (should_save_points == 'y'):
        save_pickle_points_path = input("Please enter the relative path where you would like to save to: \n").strip()
    
    should_highlight_word = input("Would you like to label the area around a specific word in your corpus? y/n \n").strip()
    if (should_highlight_word == 'y'):
        highlighted_word = input("Please enter the word: \n").strip()
    
    # Check if we should re-train the data
    if points is None:
        points = train_data(save_pickle_points_path)
    
    # Show the word, and words around it if it's found
    # Do something here with our current testing_suite.py

    print("The highlighted word is: " + str(highlighted_word))
    # Plot the data that we got here lol
    plot_data(points, highlighted_word)

    while (True):
        word_or_exit = input("Enter another word you like like to highlight or :q to quit \n")
        if (word_or_exit == ':q'):
            print("Bye!")
            exit()
        else:
            plot_data(points, word_or_exit)

if __name__ == "__main__":
    main(sys.argv)
