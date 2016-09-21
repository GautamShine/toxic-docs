import bson
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse as sp
from spacy.en import English

from sqlalchemy import create_engine
import psycopg2

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize, scale
from sklearn.cross_validation import train_test_split

"""
NLP functions (NER, lemmatization) for obtaining clean tokens
"""
class NLP():

    def __init__(self, num_chars, replace_ne=True):
        self.spacy = English()
        # only consider this many characters from beginning/end of documents
        self.num_chars = num_chars
        # replace named entities with generics
        self.replace_ne = replace_ne

    """
    Normalizes emails, people, places, and organizations
    """
    def preprocessor(self, token):

        # normalize emails
        if token.like_email:
            return 'ne_email'

        # check that the NER IOB tag is B
        if self.replace_ne and token.ent_iob == 3:

            ent = token.ent_type_

            # normalize human names
            if ent == 'PERSON':
                return 'ne_person'

            # normalize national/religious/political groups
            elif ent == 'NORP':
                return 'ne_group'

            # normalize facilities
            elif ent == 'FAC':
                return 'ne_facility'

            # normalize organizations
            elif ent == 'ORG':
                return 'ne_org'

            # normalize geopolitical places
            elif ent == 'GPE':
                return 'ne_gpe_place'
            
            # normalize natural places
            elif ent == 'LOC':
                return 'ne_nat_place'

            # normalize products
            elif ent == 'PRODUCT':
                return 'ne_product'

            # normalize laws
            elif ent == 'LAW':
                return 'ne_law'

            # normalize dates
            elif ent == 'DATE':
                return 'ne_date'

            # normalize time
            elif ent == 'TIME':
                return 'ne_time'

            # normalize percentages
            elif ent == 'PERCENT':
                return 'ne_percent'

            # normalize money
            elif ent == 'MONEY':
                return 'ne_money'

            # normalize quantity
            elif ent == 'QUANTITY':
                return 'ne_quant'

        # normalize numbers that aren't time/money/quantity/etc
        if token.is_digit:
            return 'ne_number'

        # return lemma for regular words
        return token.lemma_

    """
    Tokenizes input string with preprocessing
    """
    def tokenizer(self, doc):

        try:
            if len(doc) > 2*self.num_chars:
                spacy_doc = self.spacy(doc[:self.num_chars] + doc[-self.num_chars:])
            else:
                spacy_doc = self.spacy(doc)
                
            return [self.preprocessor(t) for t in spacy_doc \
                    if (t.is_alpha or t.like_num or t.like_email) \
                    and len(t) < 50 and len(t) > 1 \
                    and not (t.is_punct or t.is_space or t.is_stop)]
        except:
            print(doc)
            raise('Error: failed to tokenize a document')


"""
Data transformation functions to go from database dump to text features
"""
class DataProcessor():

    def __init__(self, text_key, label_key, num_chars):
        self.text_key = text_key
        self.label_key = label_key
        self.nlp = NLP(num_chars, replace_ne=True)
        self.label_dict = {'email': 0, 'internal_memo': 1,
                'boardroom_minutes': 2, 'annual_report': 3,
                'public_relations': 4, 'general_correspondance': 5,
                'media': 6,'deposition': 7,
                'scientific_article_unpublished': 8,
                'scientific_article_published': 9,
                'advertisement': 10, 'trade_association': 11,
                'contract': 12, 'budget': 13,
                'court_transcript': 14, 'general_report': 15,
                'not_english': 18, 'misc': 19, 'blank': 20}
        self.inv_label_dict = {v: k for k, v in self.label_dict.items()}
        self.label_index_list = sorted(list(self.inv_label_dict.keys()))
        self.label_name_list = sorted(self.label_dict.keys(), key=lambda x: self.label_dict[x])

    """
    Takes a bson and returns the corresponding list of dicts, the frequency of
    each label, and the number of unlabeled items
    """
    def load_bson(self, bson_file):

        # 'rb' for read as binary
        f = open(bson_file, 'rb')
        docs = bson.decode_all(f.read())

        labels = np.zeros((len(docs), 1))
        counts = defaultdict(int)

        for i in range(len(docs)):
            try:
                labels[i] = self.label_dict[docs[i][self.label_key]]
                counts[docs[i][self.label_key]] += 1
            except:
                labels[i] = -1
                counts['unlabeled'] += 1

        return docs, labels, counts

    """
    Writes the document dump (list of dicts) to a PostgresSQL table
    """
    def write_to_db(self, docs, user, pw, host, db_name):

        db = create_engine('postgres://%s:%s@%s/%s'%\
                (user, pw, host, db_name))
        conn = psycopg2.connect(database=dbname, user=user)

        df = pd.DataFrame(docs)
        df['_id'] = df['_id'].map(str)
        df.to_sql('toxic_docs_table', engine)

        conn.close()

        return

    """
    Applies a TF-IDF transformer and count vectorizer to the corpus to build
    n-gram features for classification
    """
    def vectorize(self, docs, min_df=2, max_ngram=2):

        docs = [x[self.text_key] for x in docs]
        vectorizer = TfidfVectorizer(min_df=min_df,\
                ngram_range=(1, max_ngram), tokenizer=self.nlp.tokenizer, sublinear_tf=True)

        return vectorizer, vectorizer.fit_transform(docs), vectorizer.get_feature_names()

    """
    Retrieves document features of the ToxicDocs collection
    """
    def get_feats(self, docs, key_list):

        feats = []
        for doc in docs:
            feats.append({k:v for k,v in doc.items() if k in key_list})
            try:
                feats[-1]['num_pages'] = 1+np.log(feats[-1]['num_pages'])
            except:
                pass
            feats[-1]['length'] = 1+np.log(len(doc[self.text_key]))

        vectorizer = DictVectorizer()
        X_feats = vectorizer.fit_transform(feats)
        X_feats = normalize(X_feats, axis=0, norm='max')

        return X_feats

    """
    Stacks extra features onto given data matrix
    """
    def stack_feats(self, X, feats):

        X = sp.hstack((X, feats))

        return X.tocsr()

    """
    Splits the labeled subset into train and test sets
    """
    def split_data(self, y_all, X_all, split=0.7, seed=0):
        
        indices = np.arange(y_all.shape[0])

        X_unlab = X_all[(y_all == -1).flatten()]
        ind_unlab = indices[(y_all == -1).flatten()]

        y_valid = y_all[y_all != -1]
        X_valid = X_all[(y_all != -1).flatten()]
        ind_valid = indices[(y_all != -1).flatten()]

        X_train, X_test, y_train, y_test, ind_train, ind_test =\
                train_test_split(X_valid, y_valid, ind_valid,\
                train_size=split, random_state=seed)

        return y_train, X_train, ind_train, y_test, X_test, ind_test, X_unlab, ind_unlab

    """
    Merges given classes into one label for cases when sample size is small
    """
    def merge_classes(self, merge_arr, y):

        y_merged = y.copy()
        dict_merged = {}

        for y_1, y_2 in merge_arr:
            pass

        return y_merged, dict_merged
