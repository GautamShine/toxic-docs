from flask import render_template
from flask import request
from flask_toxic import app
from sqlalchemy import create_engine
import psycopg2
import pandas as pd
import math

from credentials import *
from flask_matrix import *

db = create_engine('postgres://%s:%s@%s/%s'%(user,pw,host,dbname))
conn = psycopg2.connect(database=dbname, user=user)

@app.route('/')
@app.route('/index')
def similar_docs_input():

    return render_template('index.html')

@app.route('/similar')
def similar_docs_output():

    target_ind = request.args.get('target_ind')

    year = docs[int(target_ind)]['year']
    if type(year) is not int:
        year = 'Unknown'

    hash_id = docs[int(target_ind)]['hash_id']
    cdn_url = cdn_base+hash_id[:2]+'/'+hash_id+'/'+hash_id+'.pdf'
    target = {'index': target_ind, 'year': year, 'cdn_url': cdn_url}

    doc_inds = da.similar_docs(docs, X, target_ind, num_docs=10, print_docs=False)
    doc_inds = str(tuple(doc_inds))

    query = "SELECT index, year, document_type, hash_id FROM toxic_docs_table WHERE index IN %s" % doc_inds
    query_results=pd.read_sql_query(query, conn)

    results = []
    for i in range(0,query_results.shape[0]):

        index = query_results.iloc[i]['index']

        year = query_results.iloc[i]['year']
        if not math.isnan(float(year)):
            year = str(int(year))
        else:
            year = ''

        document_type = query_results.iloc[i]['document_type']
        if document_type is None:
            document_type = ''

        hash_id = query_results.iloc[i]['hash_id']
        cdn_url = 'http://cdn-dev.toxicdocs.org/'+hash_id[:2]+'/'+hash_id+'/'+hash_id+'.pdf'

        results.append(dict(index=index, year=year, document_type=document_type,\
                y_pred=dp.inv_label_dict[y_pred[index]], cdn_url=cdn_url))

    return render_template("similar.html", target=target, results=results)
