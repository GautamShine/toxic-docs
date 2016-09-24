#!/usr/bin/env python

from processing import NLP

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class TrendAnalyzer():

    def __init__(self, docs, num_chars):
        self.nlp = NLP(num_chars, replace_ne=False)
        self.df = pd.DataFrame(docs)

    def create_token_sets(self):

        df['tokens'] = df['text'].map(lambda x: self.nlp.set_tokenizer(x))

        return

    def word_trend(self, word, plot=False):

        # lemmatize the word
        word = self.nlp.spacy(word)[0].lemma_
        year_counts = self.df.groupby('year')['tokens'].apply(\
                lambda x: np.sum([word in y for y in x]))

        if plot:
            plt.figure()
            plt.plot(year_counts.index, year_counts.values)
            plt.show()

        return year_counts.index, year_counts.values
