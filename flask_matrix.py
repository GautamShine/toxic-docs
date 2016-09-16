#!/usr/bin/env python

from tox import *

bson_file = 'documents.bson'
label_key = 'document_type'
text_key = 'text'

# Process the raw data
# NOTE: Set num_chars high for deployment
num_chars = 50
dp = DataProcessor(text_key, label_key, num_chars)
da = DataAnalyzer(text_key)
docs, y, counts = dp.load_bson(bson_file)
_, X, _ = dp.vectorize(docs, min_df=2, max_ngram=2)

me = ModelEvaluator()
SVM = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001,\
    C=0.8, multi_class='ovr', fit_intercept=True, intercept_scaling=1,\
    class_weight='balanced', verbose=0, random_state=None, max_iter=1000)

y_lab = y[y != -1]
X_lab = X[(y != -1).flatten()]

SVM.fit(X_lab, y_lab)
SVM_y_pred, SVM_test_acc, SVM_test_prec, SVM_test_rec, SVM_test_time =\
    me.test(SVM, y_lab, X_lab)

y_pred = SVM.predict(X)
