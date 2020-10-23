import os

# data
dat_dir = os.path.join('..','data')
clean_data_path = os.path.join(dat_dir, 'complaints_2019_clean.csv.zip')
compression = 'zip'

# params
train_size = 0.8
SEED = 100
N_SAMPLES= 2_000  # small number of samples to train the model.

# models
model_dir = os.path.join('..','models')
model_linsvc_tfidf_path = os.path.join(model_dir,'tfidf.pkl')
tfidf_fitted_vec_path = os.path.join(model_dir,'tfidf_fitted_vec_path.pkl')

# output images
img_dir = os.path.join('..','images')
png_clf_report = os.path.join(img_dir, 'classification_report.png')
png_conf_mat = os.path.join(img_dir, 'confusion_matrix.png')
png_auc_roc = os.path.join(img_dir, 'auc_roc.png') # area under curve receiver operating characteristics
png_pr = os.path.join(img_dir, 'precision_recall.png')
png_cpe = os.path.join(img_dir, 'class_prediction_error.png')

# params
params_tfidf = dict(sublinear_tf=True,
                    min_df=5,
                    ngram_range=(1, 2),
                    stop_words='english')

params_linsvc = dict(C=0.728421052631579,
                    random_state=SEED)