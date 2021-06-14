import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix,accuracy_score,precision_score,recall_score 
from sklearn.model_selection import cross_val_score,GridSearchCV,train_test_split

from lib.EDA import *
from lib.models import *
from lib.utils import *
from lib.feature_analysis import *
from lib.expanded_data import *


def main():

    st.title("Warner Media Data Science Candidate Project")
    st.sidebar.title("Warner Media Data Science Candidate Project Web App")

    @st.cache(persist=True)
    def load_data():
        music = pd.read_csv("wm_project.csv")
        music_value = music.dropna().drop(['id','track','uri'],axis=1)
        return music_value
    
    data = load_data()
    
    st.sidebar.subheader("Choose Mode")
    mode = st.sidebar.selectbox("Mode", ("Exploratory Data Analysis","Model Analysis", 
                                         "Feature Analysis", "Expanded Data"))

    if mode == "Exploratory Data Analysis":
        st.sidebar.subheader("Exploratory Data Analysis")
        section = st.sidebar.selectbox("Section", ("Correlation Heatmap","Distribution Analysis", 
                                                   "Artists and Decades"))
        if section == "Correlation Heatmap":
            correlation_present(data)

        if section == "Distribution Analysis":
            eda_columns = [col for col in list(data.columns) if col not in ['hit','artist']]
            category = st.sidebar.multiselect("Plot Category",("Scatter Plot", "Histogram"))
            features = st.sidebar.selectbox("Features of Interest",tuple(eda_columns))
            if "Scatter Plot" in category:
                dist_scatter(data,features)
            if "Histogram" in category:
                dist_histogram(data,features)

        if section == "Artists and Decades":
            cols = st.sidebar.multiselect("Selected Features",("Artists", "Decades"))
            if "Artists" in cols:
                inspect_artist(data)
            if "Decades" in cols:
                inspect_decade(data)

    if mode == "Model Analysis":
        features, labels = data.drop(['hit'],axis=1),data[['hit']]

        st.sidebar.subheader("Model Development")
        stage = st.sidebar.selectbox("Stage", ("Decade Encoded", "Remove Correlation", 
                                               "Get Dummies","xgBoost Model"))
        
        st.subheader("Model Performance")
        modeling = get_model(features,labels,stage) 
        clf, X_test, y_test = modeling[0], modeling[1], modeling[2]
        accuracy = clf.score(X_test, y_test)
        y_pred = clf.predict(X_test)
        st.write("Accuracy: ", accuracy.round(5))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(5))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(5))
        try:
            feature_importance = feature_importance(clf, X_test)
            st.write("Feature Importance:\n", feature_importance)
        except:
        	pass
        plot_metrics(clf, X_test, y_test)

    if mode == "Feature Analysis":
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.subheader("Feature Analysis")
        st.sidebar.subheader("Feature Groups")
        groups = st.sidebar.multiselect("Groups of Features",('Perceptual', 'Content', 
                                                               'Physical'))
        if "Perceptual" in groups:
            learn_selected_dataset("Perceptual", data)
        if "Content" in groups:
            learn_selected_dataset("Content", data)
        if "Physical" in groups:
            learn_selected_dataset("Physical", data)

    if mode == "Expanded Data":
        st.sidebar.subheader("Analyze Expanded Dataset")
        eda_columns = [col for col in list(data.columns) if col not in ['hit','artist','sections','decade']]
        features = st.sidebar.multiselect("Features of Interest",tuple(eda_columns))
        
        extend_dataset(features)

    if st.sidebar.checkbox("Show Data Description", False):
        st.subheader("Spotify Data Set Description")
        hit = data[data['hit']==1].reset_index().drop(['index'],axis=1)
        flop = data[data['hit']==0].reset_index().drop(['index'],axis=1)
        st.write(hit.describe())
        st.write(flop.describe())


if __name__ == '__main__':
    main()
