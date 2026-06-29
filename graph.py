import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA


def bar_graph(sentences, similarity):

    scores = similarity[0]

    fig, ax = plt.subplots()

    ax.bar(range(len(scores)), scores)

    ax.set_xticks(range(len(scores)))

    ax.set_xticklabels(sentences, rotation=45)

    ax.set_ylabel("Similarity")

    st.pyplot(fig)


def heat_map(sentences, similarity):

    fig, ax = plt.subplots()

    sns.heatmap(similarity,
                annot=True,
                xticklabels=sentences,
                yticklabels=sentences,
                cmap="Blues")

    st.pyplot(fig)


def embedding_plot(sentences, embeddings):

    pca = PCA(n_components=2)

    points = pca.fit_transform(embeddings)

    fig, ax = plt.subplots()

    ax.scatter(points[:,0], points[:,1])

    for i,s in enumerate(sentences):
        ax.text(points[i,0],points[i,1],s)

    st.pyplot(fig)
