import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.svm import SVC
from scipy.stats import sem, t
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import plot_confusion_matrix, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split

import warnings
warnings.filterwarnings('ignore')

def modeling_pipeline(features, labels, model=""):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size = 0.7, random_state=0)        

    try:
        clf = joblib.load('model/{}.pkl'.format(model))
    except:        
        if model == "xgboost":
            clf = xgb.XGBClassifier(seed=0)
        else:
            clf = RandomForestClassifier(n_estimators=500,n_jobs=-1,oob_score=True,
                                    max_features='auto',random_state=0)
        clf = clf.fit(X_train,y_train)
        joblib.dump(clf, 'model/{}.pkl'.format(model))

    return clf, X_test, y_test
    
def get_model(features, labels, stage, model=""):

    if stage == "Decade Encoded":
        removed_features = ['artist']
        features = features.drop(removed_features,axis=1)
        le = LabelEncoder()
        features['decade'] = le.fit_transform(features['decade'])
        model = "encode"
    if stage == "Remove Correlation":
        removed_features = ['artist','sections']
        features = features.drop(removed_features,axis=1)
        le = LabelEncoder()
        features['decade'] = le.fit_transform(features['decade'])
        model = "correlation_removed"
    if stage == "Get Dummies":
        removed_features = ['artist','sections']
        features = features.drop(removed_features,axis=1)
        features = pd.get_dummies(features,['decade'])
        model = "dummies"
    if stage == "xgBoost Model":
        removed_features = ['artist','sections']
        features = features.drop(removed_features,axis=1)
        features = pd.get_dummies(features,['decade'])
        model = "xgboost"

    trained_model = modeling_pipeline(features, labels, model)

    return trained_model

