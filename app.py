import streamlit as st
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd

from graph import bar_graph
from graph import heat_map
from graph import embedding_plot

st.title("Text Similarity using Pretrained NLP Model")

model = SentenceTransformer("all-MiniLM-L6-v2")

text = st.text_area("Enter sentences (one per line)")

if st.button("Calculate Similarity"):

    sentences = [x for x in text.split("\n") if x!=""]

    if len(sentences) < 2:
        st.warning("Please enter at least 2 sentences.")

    else:

        embeddings = model.encode(sentences)

        similarity = util.cos_sim(embeddings, embeddings).numpy()

        st.subheader("Similarity Matrix")

        st.dataframe(pd.DataFrame(similarity,
                                  index=sentences,
                                  columns=sentences))

        bar_graph(sentences, similarity)

        heat_map(sentences, similarity)

        embedding_plot(sentences, embeddings)

        st.header("Paul Critical Thinking")

        st.write("### Clarity")
        st.write("Input sentences were compared using semantic similarity.")

        st.write("### Accuracy")
        st.write("Model: all-MiniLM-L6-v2")

        st.write("### Precision")
        st.write("Exact cosine similarity scores are displayed.")

        st.write("### Relevance")
        st.write("Graphs directly support similarity results.")

        st.write("### Logic")
        st.write("Higher score means higher semantic similarity.")

        st.write("### Significance")
        st.write("Highest similarity pair is the most related.")

        st.write("### Fairness")
        st.write("Model may not understand every domain-specific sentence.")
