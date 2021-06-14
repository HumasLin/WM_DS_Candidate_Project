import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter

# inspect the correlation between features
def correlation_present(data):
    st.subheader("Correlation Heatmap")
    fig = plt.figure(figsize=(12,6), dpi=500)
    cols = data.corr()['hit'].index
    correlations = np.abs(np.corrcoef(data[cols].values.T))
    sns.heatmap(correlations, square=True, yticklabels=cols.values, xticklabels=cols.values, 
                annot=True, annot_kws={"fontsize":5}, cmap="Reds")
    st.pyplot(fig)

# View distribution of data
# By scatter plot
def dist_scatter(data, col):
    st.subheader("Scatter Plot")
    fig, ax = plt.subplots()
    plt.scatter(data[col],data['hit'])
    plt.xlabel(col)
    plt.yticks([0,1],['flop','hit'])
    plt.show()
    st.pyplot(fig)

# By histogram
def dist_histogram(data, col):
    st.subheader("Histogram")
    fig, ax = plt.subplots()
    hit = data[data['hit']==1].reset_index().drop(['index'],axis=1)
    flop = data[data['hit']==0].reset_index().drop(['index'],axis=1)
    sns.distplot(hit[col], kde_kws={"color": "red", "linestyle": "-"}, hist_kws={"color": "dodgerblue"})
    sns.distplot(flop[col], kde_kws={"color": "yellow", "linestyle": "-"}, hist_kws={"color": "black"})
    plt.xlabel(col)
    plt.ylabel("Frequency (normalized)")
    plt.show()
    st.pyplot(fig)

# Inspect artist
def inspect_artist(data):
    st.subheader("Inspect Artists")
    hit = data[data['hit']==1].reset_index().drop(['index'],axis=1)
    flop = data[data['hit']==0].reset_index().drop(['index'],axis=1)
    
    st.markdown("Top 10 Hit Artist")
    artist_hit = sorted(Counter(hit['artist']).items(), key=lambda x:x[1], reverse=True)
    message_hit = "; ".join([line[0]+': '+str(line[1]) for line in artist_hit[:10]])
    st.write(message_hit)
    st.markdown("Top 10 Flop Artist")
    artist_flop = sorted(Counter(flop['artist']).items(), key=lambda x:x[1], reverse=True)
    message_flop = "; ".join([line[0]+': '+str(line[1]) for line in artist_flop[:10]])
    st.write(message_flop)

    artist_record = {}
    count_hit = Counter(hit['artist'])
    count_flop = Counter(flop['artist'])
    for artist in data['artist']:
        n_hit = count_hit[artist]
        n_flop = count_flop[artist]
        artist_record[artist]=[n_hit,n_flop]
    artist_cor = np.array(list(artist_record.values()))

    fig = plt.figure(figsize=(12,6), dpi=200)

    plt.scatter(artist_cor[:,0],artist_cor[:,1])
    plt.xlabel('flop records')
    plt.ylabel('hit records')
    plt.show()
    st.pyplot(fig)

def inspect_decade(data):
    st.subheader("Inspect Decades")
    hit = data[data['hit']==1].reset_index().drop(['index'],axis=1)
    flop = data[data['hit']==0].reset_index().drop(['index'],axis=1)

    decade_record = {}
    count_hit = Counter(hit['decade'])
    count_flop = Counter(flop['decade'])

    for decade in data['decade']:
        n_hit = count_hit[decade]
        n_flop = count_flop[decade]
        decade_record[decade]=[n_hit,n_flop]

    decade_cor = np.array(list(decade_record.values()))
    decade_name = list(decade_record.keys())

    fig = plt.figure(figsize=(10,6), dpi=200)

    plt.scatter(decade_cor[:,0],decade_cor[:,1])
    plt.xlabel('flop records')
    plt.ylabel('hit records')
    for i, txt in enumerate(decade_name):
        plt.annotate(txt, (decade_cor[:,0][i], decade_cor[:,1][i]))
    plt.show()
    st.pyplot(fig)

