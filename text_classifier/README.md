# Text classifier

Implements a text classifier using a *bag-of-words* data representation. The model is built using a training dataset with roughly 11,000 excerpts from documents labeld with 8 different categories. The test data consists of rouglhy 2,200 of such excerpts

The data can be downloaded from:
https://lazyprogrammer.me/course_files/deepnlp_classification_data.zip

An object of `TextClassifier` class can be instantiated either with a GloVe or a Word2Vec embedding. Using the embedding the feature vector is built as
```math
feature = \frac{1}{|document|}\sum_{w \in document} vec(w)
```
