#!/usr/bin/env python

"""
__author__ = 'Gautam Shine'
__email__ = 'gshine@stanford.edu'

Word trend analyzer for the "Toxic Docs" repository from Columbia University
and the Center for Public Integrity. Data set consists of PDF files of
emails, memos, advertisements, news articles, scientific articles cited in
legal cases involving allegations of environmental harm from toxic substances.

"""

from processing import *
from modeling import *
from analyzing import *

import time
from sklearn.svm import LinearSVC

class TrendAnalyzer():

    def __init__(self, docs, num_chars=1000, init_now=False):

        self.nlp = NLP(num_chars, replace_ne=False)
        self.df = pd.DataFrame(docs)

        # perform expensive parts of initialization
        if init_now:
            self.create_token_sets()
            self.infer_doc_years()

    """
    Uses the set tokenizer to retrieve unique words
    """
    def create_token_sets(self):

        self.df['tokens'] = self.df['text'].map(lambda x: self.nlp.set_tokenizer(x))

        return
    
    """
    Infers a year by taking of the max of year-like tokens
    """
    def infer_year(self, tokens):
    
        years = []
        for token in tokens:
            try:
                if self.nlp.spacy(token)[0].is_digit:
                    try:
                        num = int(token)
                        if num < 2016 and num > 1890:
                            years.append(num)
                    except:
                        pass
            except:
                pass
        if years:
            return int(max(years))
        else:
            return None

    """
    Stores inferred year of documents in data frame
    """
    def infer_doc_years(self):

        self.df['inferred_year'] = self.df['tokens'].map(lambda x: self.infer_year(x))

        return

    """
    Computes a histogram of word appearence by year
    """
    def word_trend(self, word, plot=False):

        # lemmatize the word
        word = self.nlp.spacy(word)[0].lemma_
        
        # compute series of appearance counts
        year_counts = self.df.groupby(['inferred_document_type', 'inferred_year'])\
                ['tokens'].apply(lambda x: np.sum([word in y for y in x]))

        if plot:
            
            plt.figure(figsize=(10,6))
            doc_type = [1,6,8,9]
            colors=['-r', '-g', '-b', '-k']
            max_count = 0
            
            for i,dt in enumerate(doc_type):
                try:
                    x = year_counts[dt].index
                    y = year_counts[dt].values
                    if np.amax(y) > max_count:
                        max_count = np.amax(y)
                    plt.plot(x, y, colors[i])
                except:
                    pass

            axes = plt.gca()
            axes.set_xlim([1940,2016])
            axes.set_ylim([0,max_count*1.2])            
            plt.show()
            
        return year_counts
        
    """
    Computes indices of most similar documents in n-gram space
    """
    def compute_sim_docs(self, X, num_docs):
    
        self.sim_docs = np.zeros((X.shape[0], num_docs), dtype=int)
        for i in range(X.shape[0]):
            doc_sims = np.dot(X, X[i, :].transpose())
            sim_ranking = np.argsort(-doc_sims.toarray().flatten())
            self.sim_docs[i,:] = sim_ranking[1:(num_docs+1)]


if __name__ == '__main__':

    bson_file = 'doc_extra.bson'
    label_key = 'document_type'
    text_key = 'text'

    # Process the raw data
    dp = DataProcessor(text_key, label_key, num_chars=3, replace_ne=True)
    da = DataAnalyzer(text_key)
    docs, y_regex, counts_regex = dp.load_bson(bson_file)
    ta = TrendAnalyzer(docs, num_chars=10, init_now=True)

    t0 = time.time()
    ta.create_token_sets()
    tok_time = time.time() - t0
    print('Tokenization time:', tok_time)

    t0 = time.time()
    vectorizer, X_all_ngram, feat_names = dp.vectorize(docs, min_df=5, max_ngram=2)
    vec_time = time.time() - t0

    # Replace regex labels with human labels
    y_all = np.loadtxt('labels.txt', dtype=np.int32)

    # Merge similar classes
    y_merged = dp.merge_classes([(7,14), (15,19), (15, 12), (15, 18)], y_all)
    counts = np.bincount(y_merged[y_merged != -1])
    counts = [counts[i] for i in range(len(counts)) if i in dp.label_index_list]
    print(counts)

    # Add extra features from ToxicDocs to n-gram data matrix
    key_list = ['num_pages']
    feats = dp.get_feats(docs, key_list)
    X_all = dp.stack_feats(X_all_ngram, feats)
    key_list.append('length')
    feat_names.extend(key_list)

    print('Vectorization time:', vec_time)
    print('Data matrix size:', X_all.shape)

    y_train, X_train, ind_train, y_test, X_test, ind_test, X_unlab, ind_unlab =\
            dp.split_data(y_merged, X_all, split=0.7, seed=0)
    me = ModelEvaluator()

    # LinearSVC (liblinear SVM implementation, one-v-all)
    cross_validate = True
    if cross_validate:
        model = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001,\
                C=1, multi_class='ovr', fit_intercept=True, intercept_scaling=1,\
                class_weight='balanced', verbose=0, random_state=None, max_iter=1000)
        param_grid = {'C':np.logspace(-1,1,24).tolist()}
        grid_info, grid_best, grid_time = me.param_search(model, param_grid,\
                np.concatenate((y_train, y_test)), sp.vstack((X_train, X_test)), num_folds=3)
        C = grid_best['C']
    else:
        C = 1
    print(C)

    SVM = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001,\
            C=C, multi_class='ovr', fit_intercept=True, intercept_scaling=1,\
            class_weight='balanced', verbose=0, random_state=None, max_iter=1000)

    SVM_train_acc, SVM_train_time = me.train(SVM,\
            np.concatenate((y_train, y_test)), sp.vstack((X_train, X_test)))

    # Perform semisupervised learning
    ssl = SemiSupervisedLearner(SVM)
    ave_conf = np.mean(ssl.confidence_scores(SVM.decision_function(\
            sp.vstack((X_train, X_test)))))
    print(ave_conf)

    y_working = ssl.loop_learning(X_unlab,\
            np.concatenate((y_train, y_test)), sp.vstack((X_train, X_test)),\
            dp.label_index_list,num_loops=1, conf_thresh=1.6*ave_conf)

    y_pred = ssl.model.predict(X_all)
    ta.df['inferred_document_type'] = y_pred

    print(time.time() - t0)
    ta.df['inferred_year'].count()

    compute_sim = False
    if compute_sim:
        t0 = time.time()
        ta.compute_sim_docs(X_all, 100)
        sim_time = time.time() - t0
        print('Similarity computation time', sim_time)

    wt = ta.word_trend('vinyl', plot=True)
