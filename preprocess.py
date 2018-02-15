import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
import sys
import time
import random
from collections import defaultdict
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy import sparse
import xgboost as xgb

import preprcessUtil

utils = preprcessUtil.Utils()

FEATURE_NOT_USE = ['bookname']  #



def bookNameProcess(data, train_idx, test_idx):
    data["bookname_num"] = data["bookname"].apply(lambda x: len(x.split(' ')))
    data["bookname_len"] = data["bookname"].apply(len)
    '''
    bookNameVector = vectorizer.fit_transform(data['bookname']).toarray()
    TfidfVectorizer(min_df=20, max_features=50, strip_accents='unicode', 
        lowercase=True, analyzer='word', token_pattern=r'\w{16,}', ngram_range=(1, 2), 
        use_idf=False, smooth_idf=False, sublinear_tf=True, stop_words='english')
    '''
    tfidfdesc = TfidfVectorizer(stop_words='english')
    tr_sparsed = tfidfdesc.fit_transform(data.iloc[train_idx, :]["bookname"])
    te_sparsed = tfidfdesc.transform(data.iloc[test_idx, :]["bookname"])
    feats_names = ["desc_" + x for x in tfidfdesc.get_feature_names()]
    return data, tr_sparsed, te_sparsed, feats_names



def coreProcess(data, y_train, train_idx, test_idx):
    data['fnumber'] = data['fnumber'].apply(utils.processfNumber)
    data['temperature'] = data['temperature'].apply(utils.processTemp)
    data['rainfall'] = data['rainfall'].apply(utils.processRain)
    data['bookpages'] = data['bookpages'].apply(utils.processBook)

    data['fnumber'].fillna(data['fnumber'].mode(), inplace=True)
    data['temperature'].fillna(data['temperature'].mode(), inplace=True)
    data['rainfall'].fillna(data['rainfall'].mode(), inplace=True)
    data.fillna('0', inplace=True)
    data, tr_sparsed, te_sparsed, feats_sparsed = bookNameProcess(data, train_idx, test_idx)




    #print (data[["music", "timestamp"]].groupby(['Sex'], as_index=False).mean())
    #print data.info()

    #print data['comic'].value_counts()


    categorical = ['fnumber', 'temperature', 'rainfall', 'bookpages', 'comic', 'music']
    for f in categorical:
        if data[f].dtype == 'object':
            cases = defaultdict(int)
            temp = np.array(data[f]).tolist()
            for k in temp:
                cases[k] += 1
            #print(f, len(cases))
            data[f] = data[f].apply(lambda x: cases[x])

    #data.to_csv('temp.csv',index=False)


    #print (data[['fnumber', 'timestamp']].groupby(['fnumber'], as_index=False).mean())
    #print (data[["temperature", "timestamp"]].groupby(['temperature'], as_index=False).mean())
    #print (data[["rainfall", "timestamp"]].groupby(['rainfall'], as_index=False).mean())
    #print (data[["bookpages", "timestamp"]].groupby(['bookpages'], as_index=False).mean())



    feats_in_use = [col for col in data.columns if col not in FEATURE_NOT_USE]

    data_train = np.array(data.iloc[train_idx, :][feats_in_use])
    data_test = np.array(data.iloc[test_idx, :][feats_in_use])

    stda = StandardScaler()
    data_test = stda.fit_transform(data_test)
    data_train = stda.transform(data_train)

    data_train = sparse.hstack([data_train, tr_sparsed]).tocsr()
    data_test = sparse.hstack([data_test, te_sparsed]).tocsr()
    feats_in_use.extend(feats_sparsed)

    '''
    colormap = plt.cm.viridis
    plt.figure(figsize=(12, 12))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    g = sns.pairplot(data_train[data.columns])
    g.set(xticklabels=[])
    '''

    return data_train, data_test, feats_in_use









