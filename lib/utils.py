import numpy as np 
import streamlit as st
from scipy.stats import sem, t
from sklearn.metrics import plot_confusion_matrix,plot_roc_curve,plot_precision_recall_curve

class_names = ['flop','hit']

def conf_int(scores):

    confidence = 0.95
    data = scores

    n = len(data)
    m = np.mean(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)

    start = m - h
    end = m + h

    return([start,end])

def feature_importance(clf, features):

    update_columns = list(features.columns)
    res = clf.feature_importances_
    zip_list = dict(zip(update_columns,res)).items()

    sorted_importances = sorted(zip_list,key=lambda x:x[1],reverse=True)
    message = "\n".join([line[0]+": "+str(np.round(line[1],3)) for line in sorted_importances])
    return message

def plot_metrics(model,X_test,y_test):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.subheader("Confusion Matrix") 
    plot_confusion_matrix(model, X_test, y_test, display_labels=class_names)
    st.pyplot()
    
    st.subheader("ROC Curve") 
    plot_roc_curve(model, X_test, y_test)
    st.pyplot()

    st.subheader("Precision-Recall Curve")
    plot_precision_recall_curve(model, X_test, y_test)
    st.pyplot()