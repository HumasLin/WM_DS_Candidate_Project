import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix,accuracy_score,precision_score,recall_score

features_dict = {'Perceptual':['danceability', 'energy', 'valence'],
                 'Content':['speechiness', 'acousticness', 'instrumentalness', 'liveness'],
                 'Physical':['tempo','key','loudness','duration_ms','time_signature','chorus_hit','sections']
                }

def learn_selected_dataset(key, data):
    features = features_dict[key]
    selected_data = data[features]
    hit_label = data[['hit']]
    X_train, X_test, y_train, y_test = train_test_split(selected_data, hit_label, train_size = 0.7, random_state=0)
    
    try:
        clf = joblib.load('model/{}_features.pkl'.format(key.lower()))
    except:
        clf = xgb.XGBClassifier(seed=0)
        clf = clf.fit(X_train,y_train)
        joblib.dump(clf, 'model/{}_features.pkl'.format(key.lower()))
    
    y_pred = clf.predict(X_test)
    
    st.subheader(key + " Features Performance:") 
    st.write("Accuracy: ",np.round(accuracy_score(y_pred,y_test),5))
    st.write("Precision: ", precision_score(y_test, y_pred).round(5))
    st.write("Recall: ", recall_score(y_test, y_pred).round(5))
    plot_confusion_matrix(clf, X_test, y_test, values_format='d')
    st.pyplot()