import numpy as np
from gensim.models import KeyedVectors
import sys
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class TextClassifier:

    def __init__(self, embedding='glove', classifier=LogisticRegression()):

        # Load word embedding
        self.embedding = embedding
        if self.embedding == 'glove':
            self.embedding_dim = 50
            self.word_vectors = {}
            with open('embeddings/glove.6B/glove.6B.50d.txt') as f:
                for line in f:
                    line_list = line.split()
                    word = line_list[0]
                    word_vec = np.asarray(line_list[1:], dtype=np.float32)
                    self.word_vectors[word] = word_vec
        elif self.embedding == 'word2vec':
            self.embedding_dim = 300
            self.word_vectors = KeyedVectors.load_word2vec_format('embeddings/GoogleNews-vectors-negative300.bin',
                                                                  binary=True)
        else:
            sys.exit("Invalid embedding specification. Please choose embedding in ['glove', 'word2vec']")

        # Load data
        self.train = pd.read_csv('text_classifier/deepnlp_classification_data/r8-train-all-terms.txt', header=None,
                                 sep='\t')
        self.test = pd.read_csv('text_classifier/deepnlp_classification_data/r8-test-all-terms.txt', header=None,
                                sep='\t')
        self.train.columns = ['label', 'content']
        self.test.columns = ['label', 'content']
        self.train['content'] = [self.train['content'].values[i].split() for i in range(len(self.train.index))]
        self.test['content'] = [self.test['content'].values[i].split() for i in range(len(self.test.index))]

        # Build features
        def build_feature(word_list):
            feature = np.zeros(self.embedding_dim)
            n = len(word_list)
            for word in word_list:
                if word in self.word_vectors:
                    feature += self.word_vectors[word]
                else:
                    n -= 1
            return (feature / n).tolist()

        self.train['feature'] = self.train.apply(lambda row: build_feature(row['content'], ), axis=1)
        self.test['feature'] = self.test.apply(lambda row: build_feature(row['content']), axis=1)

        self.train_features = self.train['feature']
        self.train_features = pd.DataFrame.from_dict(dict(zip(self.train_features.index, self.train_features.values))).T
        self.test_features = self.test['feature']
        self.test_features = pd.DataFrame.from_dict(dict(zip(self.test_features.index, self.test_features.values))).T

        # Set up the model
        self.model = classifier

    def fit(self):
        self.model.fit(self.train_features, self.train['label'])

    def get_training_score(self):
        return self.model.score(self.train_features, self.train['label'])

    def get_test_score(self):
        return self.model.score(self.test_features, self.test['label'])


if __name__ == '__main__':

    embeddings = ['glove', 'word2vec']
    classifiers = [RandomForestClassifier(), LogisticRegression(), XGBClassifier()]
    for embedding in embeddings:
        for classifier in classifiers:
            classifier_string = str(classifier)
            classifier_string = classifier_string[:classifier_string.find('(')]
            text_classifier = TextClassifier(embedding, classifier)
            text_classifier.fit()
            print(f"%%%%%% Using {embedding} and {classifier_string} we get")
            print(f"\t Training score: {text_classifier.get_training_score():.4f}")
            print(f"\t Test score: {text_classifier.get_test_score():.4f}")
            del text_classifier
