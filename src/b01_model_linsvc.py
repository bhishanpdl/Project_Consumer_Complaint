# Load the libraries
#================================================================
import time
time_start_notebook = time.time()

import numpy as np
import pandas as pd
import sklearn
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib

import util
import config

# params
RE_TRAIN = False

ifile = config.clean_data_path
SEED = config.SEED
model_linsvc_tfidf_path = config.model_linsvc_tfidf_path
tfidf_fitted_vec_path = config.tfidf_fitted_vec_path

compression= config.compression

# Load the data
#================================================================
usecols = ['complaint_clean','product']
df = pd.read_csv(ifile,compression=compression,usecols=usecols)
print(f'clean data shape: {df.shape}')

# Data Processing
#==============================================
X = df['complaint_clean'] # documents
y = df['product'].astype('category').cat.codes # target

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=config.train_size,
                                                    random_state=config.SEED)

# Modelling
#===============================================================
if RE_TRAIN:
    # vectorizer and model
    tfidf = TfidfVectorizer(**config.params_tfidf)
    model = svm.LinearSVC(**config.params_linsvc)

    # fit the vectorizer
    fitted_vectorizer = tfidf.fit(X_train)
    Xtrain_text = fitted_vectorizer.transform(X_train)

    # fit the model
    model.fit(Xtrain_text, y_train)

    # persist the models
    joblib.dump(model, model_linsvc_tfidf_path )
    joblib.dump(fitted_vectorizer, tfidf_fitted_vec_path)

#==================================================================
# Model Evaluation
fitted_vectorizer = joblib.load(tfidf_fitted_vec_path)
model = joblib.load(model_linsvc_tfidf_path)

X_test = fitted_vectorizer.transform(X_test)
ypreds = model.predict(X_test)
print('Accuracy              : {:.4f} '.format(metrics.accuracy_score(y_test,ypreds)))

print(metrics.classification_report(y_test, ypreds,
    target_names= df['product'].unique()))

# Time Taken
time_taken = time.time() - time_start_notebook
util.print_time_taken(time_taken)