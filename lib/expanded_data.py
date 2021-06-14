import joblib
import numpy as np 
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from lib.utils import *

def extend_dataset(features):
    music = pd.read_csv("wm_project.csv")

    index_hit_isnan = music.isnull().any(axis=1)
    music_w_nan = music[index_hit_isnan].reset_index().drop(['index'],axis=1)

    music_w_nan = music_w_nan.drop(['id','track','uri'],axis=1)
    music_w_nan = music_w_nan.drop(['artist','sections','hit'],axis=1)

    music_w_nan = pd.get_dummies(music_w_nan,['decade'])
    clf_xgb = joblib.load("best_model.pkl")

    music_w_nan['hit'] = clf_xgb.predict(music_w_nan)

    df_wo_nan = music.dropna().drop(['id','track','uri'],axis=1)
    removed_features = ['artist','sections','hit']
    music_wo_nan = df_wo_nan.drop(removed_features,axis=1)
    music_wo_nan = pd.get_dummies(music_wo_nan,['decade'])
    music_wo_nan['hit'] = df_wo_nan['hit']

    complete_data = pd.concat([music_wo_nan,music_w_nan])
    complete_data = complete_data.reset_index().drop(['index'],axis=1)

    new_hit = complete_data[complete_data['hit']==1]
    new_flop = complete_data[complete_data['hit']==0]
    st.markdown("Model used: Trained xgBoost model")
    st.markdown("Hit songs in Expanded Dataset: {}".format(len(new_hit)))
    st.markdown("Flop songs in Expanded Dataset: {}".format(len(new_flop)))

    st.subheader("Analysis on Expanded Dataset")
    for feature in features:
        st.markdown("{}:".format(feature))
        fig = plt.figure(figsize=(9,2), dpi=200)

        ci_hit = conf_int(new_hit[feature])        
        ci_flop = conf_int(new_flop[feature])
        ci_left = [ci_hit[0],ci_flop[0]]
        ci_right = [ci_hit[1],ci_flop[1]]

        left = min(min(ci_left)-0.05*abs(min(ci_left)),max(ci_left)-0.1*max(ci_right))
        right = max(ci_right)+abs(0.1*max(ci_right))

        plt.plot(ci_hit,[1,1],label="Hit",c="red")
        plt.plot(ci_flop,[0,0],label="Flop",c="blue")
        plt.xlim(left,right)
        plt.xticks(np.linspace(left,right,5))
        plt.xlabel("Confidence Interval")
        plt.yticks([0,1],['flop','hit'])
        plt.legend()
        st.pyplot(fig)
