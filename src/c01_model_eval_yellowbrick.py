
# time
import time
time_start_notebook = time.time()

# local scripts
import util
import config

ifile = config.clean_data_path
model_linsvc_tfidf_path = config.model_linsvc_tfidf_path
tfidf_fitted_vec_path = config.tfidf_fitted_vec_path
compression= config.compression
SEED = config.SEED
N_SAMPLES = config.N_SAMPLES

png_clf_report = config.png_clf_report
png_conf_mat = config.png_conf_mat
png_auc_roc = config.png_auc_roc # area under curve receiver operating characteristics
png_pr = config.png_pr # precision-recall
png_cpe = config.png_cpe # class prediction error

# usual imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib

# Visualizers
import yellowbrick
from yellowbrick import classifier as yclf

#===========================================================================
# Load the data
df = pd.read_csv('../data/complaints_2019_clean.csv.zip',compression='zip')
df = df.sample(n=N_SAMPLES, random_state=SEED)


#============================================================================
# Data preparation
df['product_id'] = df['product'].astype('category').cat.codes
X = df['complaint_clean'] # documents
y = df['product_id'] # target
classes = df['product'].unique()

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=config.train_size,
                                                    random_state=config.SEED)

y_train = np.array(y_train).flatten()
y_test = np.array(y_test).flatten()

model = joblib.load(model_linsvc_tfidf_path)
fitted_vectorizer = joblib.load(tfidf_fitted_vec_path)

X_train = fitted_vectorizer.transform(X_train)
X_test = fitted_vectorizer.transform(X_test)

#=============================================================================
def viz_metrics(visualizer,outpath=None,
                 Xtr=X_train,Xtx=X_test,
                 ytr=y_train,ytx=y_test):

    visualizer.fit(Xtr, ytr)
    visualizer.score(Xtx, ytx)
    visualizer.poof(outpath=outpath)
    plt.close()

# classification report
fig,ax = plt.subplots(figsize=(12,8))
visualizer = yclf.ClassificationReport(model, classes=classes, support=True, ax=ax)
viz_metrics(visualizer,png_clf_report)

# confusion matrix
fig,ax = plt.subplots(figsize=(12,8))
visualizer = yclf.ConfusionMatrix(model, classes=classes,percent=True, ax=ax)
viz_metrics(visualizer, png_conf_mat)

# roc auc
fig,ax = plt.subplots(figsize=(12,8))
visualizer = yclf.ROCAUC(model, classes=classes, ax=ax)
viz_metrics(visualizer, png_auc_roc)

# precision-recall
fig,ax = plt.subplots(figsize=(12,8))
visualizer = yclf.PrecisionRecallCurve(model,classes=classes,per_class=True,
                iso_f1_curves=False,fill_area=False, micro=False, ax=ax)
viz_metrics(visualizer, png_pr)

# class prediction error
fig,ax = plt.subplots(figsize=(12,8))
visualizer = yclf.ClassPredictionError(model, classes=classes, ax=ax)
viz_metrics(visualizer, png_cpe)
