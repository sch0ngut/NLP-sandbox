# Text classifier

Implements a text classifier using a *bag-of-words* data representation. The model is built using a training dataset with roughly 11,000 excerpts from documents labeld with 8 different categories. The test data consists of rouglhy 2,200 of such excerpts

The data can be downloaded from:
https://lazyprogrammer.me/course_files/deepnlp_classification_data.zip

An object of `TextClassifier` class can be instantiated either with a GloVe or a Word2Vec embedding and using any classifier (e.g. RandomForest).

Here are the training / test accuracies of some selected classifiers and both embeddings:

   -    | Logistic Regression | Random Forest | XGBoost       |
---     | ------------------- | ------------- | ------------- |
GloVe   |  93.91% / 94.20%    |99.93% / 92.96%|99.93% / 93.88%|
Word2vec|  95.13% / 94.15%    |99.93% / 93.83%|99.93% / 95.61%|
