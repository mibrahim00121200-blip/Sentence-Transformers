import streamlit as st
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd

# Ensure graph.py exists in your root directory!
from graph import bar_graph
from graph import heat_map
from graph import embedding_plot

st.title("Text Similarity using Pretrained NLP Model")

# FIX 1: Cache the model so it doesn't reload/crash on every single button click
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

text = st.text_area("Enter sentences (one per line)")

if st.button("Calculate Similarity"):
    
    # FIX 2: Use splitlines() and strip() to handle Windows '\r' and accidental blank spaces
    sentences = [x.strip() for x in text.splitlines() if x.strip()]

    if len(sentences) < 2:
        st.warning("Please enter at least 2 sentences.")
        
    # FIX 3: Prevent Pandas from crashing if user enters duplicate sentences
    elif len(sentences) != len(set(sentences)):
        st.error("Please ensure all entered sentences are unique. Duplicate rows/columns break the matrix visualization.")

    else:
        with st.spinner("Calculating embeddings..."):
            embeddings = model.encode(sentences)

            # FIX 4: .cpu().numpy() prevents crashes if the model runs on a GPU (CUDA)
            similarity = util.cos_sim(embeddings, embeddings).cpu().numpy()

            st.subheader("Similarity Matrix")

            st.dataframe(pd.DataFrame(similarity,
                                      index=sentences,
                                      columns=sentences))

            # Make sure these functions in your graph.py accept numpy arrays properly!
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
