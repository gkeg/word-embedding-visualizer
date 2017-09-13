# harry_potter_word_vectors
Visualization of Words in Harry Potter using Word2Vec

This project contains two key components. 

1. The first is a data folder which contains every single book of Harry Potter in a .txt format.
    * You can fill this folder with any text files you want and Word2Vec will analyze those words instead. However, it has been filled with the Harry Potter books for the purposes of this project
2. The second is an executable file that will read in all of the files, and use the data to train Word2Vec. Then, the data is compressed from 250-dimensional space to 2-dimensions using T-SNE and visualized with Seaborn and MatPlotLib.

**Zoomed In Example:**
![Visualized Vectors](https://github.com/gkeglevich/harry_potter_word_vectors/blob/master/Screenshots/Figure%201.png "Example")
