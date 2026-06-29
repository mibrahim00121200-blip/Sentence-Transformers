import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Text Similarity Explorer",
    page_icon="🧠",
    layout="wide"
)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🧠 NLP Text Similarity Explorer")
st.markdown("**Model:** `all-MiniLM-L6-v2` (Free Pretrained — No training, No preprocessing)")
st.divider()

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()
st.success("✅ Model loaded: all-MiniLM-L6-v2 (Sentence Transformers)")

# ── Input Section ─────────────────────────────────────────────────────────────
st.subheader("📝 Enter Your Texts")
col1, col2 = st.columns(2)

with col1:
    query = st.text_area(
        "Query Text (the text you want to compare against):",
        value="Artificial intelligence is transforming the world.",
        height=100
    )

with col2:
    candidates_raw = st.text_area(
        "Candidate Texts (one per line — at least 4):",
        value=(
            "Machine learning is a subset of AI.\n"
            "Deep learning uses neural networks.\n"
            "The stock market crashed yesterday.\n"
            "Natural language processing helps computers understand text.\n"
            "Cats are wonderful pets.\n"
            "Robots can perform complex tasks using AI."
        ),
        height=100
    )

run = st.button("🚀 Compute Similarity", type="primary")

# ── Computation & Graphs ──────────────────────────────────────────────────────
if run:
    candidates = [c.strip() for c in candidates_raw.strip().split("\n") if c.strip()]

    if len(candidates) < 2:
        st.error("Please enter at least 2 candidate texts.")
        st.stop()

    all_texts = [query] + candidates

    # Encode — model does this internally; no manual preprocessing
    with st.spinner("Computing embeddings…"):
        embeddings = model.encode(all_texts)

    query_emb   = embeddings[0:1]
    cand_embs   = embeddings[1:]

    # Similarity scores
    scores = cosine_similarity(query_emb, cand_embs)[0]
    sorted_idx = np.argsort(scores)[::-1]
    sorted_candidates = [candidates[i] for i in sorted_idx]
    sorted_scores     = scores[sorted_idx]

    # Full pairwise matrix (all texts)
    pairwise = cosine_similarity(embeddings)

    st.divider()
    st.subheader("📊 Results")

    # ── Similarity table ──────────────────────────────────────────────────────
    st.markdown("#### Similarity Scores (Query vs Each Candidate)")
    for i, (txt, sc) in enumerate(zip(sorted_candidates, sorted_scores)):
        st.markdown(f"**{i+1}.** `{sc:.4f}` — {txt}")

    st.divider()

    # ── Graph 1 : Bar Chart ───────────────────────────────────────────────────
    st.subheader("📈 Graph 1 — Bar Chart: Similarity Scores")
    short_labels = [f"C{sorted_idx[i]+1}" for i in range(len(sorted_candidates))]

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    colors = plt.cm.RdYlGn(sorted_scores)
    bars = ax1.barh(short_labels[::-1], sorted_scores[::-1], color=colors[::-1])
    ax1.set_xlabel("Cosine Similarity Score", fontsize=12)
    ax1.set_title("Top Similar Candidates to Query Text", fontsize=14, fontweight="bold")
    ax1.set_xlim(0, 1)
    for bar, sc in zip(bars, sorted_scores[::-1]):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{sc:.4f}", va="center", fontsize=10)
    # Legend mapping
    legend_text = "\n".join([f"C{sorted_idx[i]+1} = {sorted_candidates[i][:45]}…"
                              if len(sorted_candidates[i]) > 45
                              else f"C{sorted_idx[i]+1} = {sorted_candidates[i]}"
                              for i in range(len(sorted_candidates))])
    ax1.text(1.02, 0.5, legend_text, transform=ax1.transAxes,
             fontsize=7, va="center", bbox=dict(boxstyle="round", fc="lightyellow"))
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

    st.divider()

    # ── Graph 2 : Heatmap ─────────────────────────────────────────────────────
    st.subheader("🌡️ Graph 2 — Heatmap: Pairwise Similarity")
    heat_labels = ["Query"] + [f"C{i+1}" for i in range(len(candidates))]

    fig2, ax2 = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        pairwise,
        annot=True, fmt=".2f",
        xticklabels=heat_labels, yticklabels=heat_labels,
        cmap="YlOrRd", vmin=0, vmax=1, ax=ax2,
        linewidths=0.5
    )
    ax2.set_title("Pairwise Cosine Similarity Matrix", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # Candidate legend below heatmap
    st.markdown("**Legend:**")
    for i, txt in enumerate(candidates):
        st.markdown(f"- **C{i+1}** → {txt}")

    st.divider()

    # ── Graph 3 : 2D PCA Embedding Plot ──────────────────────────────────────
    st.subheader("🗺️ Graph 3 — 2D PCA Embedding Plot")
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(embeddings)

    fig3, ax3 = plt.subplots(figsize=(9, 6))
    # Query point
    ax3.scatter(reduced[0, 0], reduced[0, 1], s=200, c="red",
                zorder=5, marker="*", label="Query")
    ax3.annotate("Query", (reduced[0, 0], reduced[0, 1]),
                 textcoords="offset points", xytext=(8, 5), fontsize=10,
                 color="red", fontweight="bold")
    # Candidate points
    scatter_colors = plt.cm.tab10(np.linspace(0, 1, len(candidates)))
    for i, (x, y) in enumerate(reduced[1:]):
        ax3.scatter(x, y, s=100, color=scatter_colors[i], zorder=4)
        ax3.annotate(f"C{i+1}", (x, y), textcoords="offset points",
                     xytext=(6, 4), fontsize=9)

    ax3.set_title("2D PCA — Semantic Space of All Texts", fontsize=14, fontweight="bold")
    ax3.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax3.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax3.legend()
    ax3.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    st.divider()

    # ── Paul's Critical Thinking Standards ───────────────────────────────────
    st.subheader("🧩 Paul's Critical Thinking Standards — Analysis")
    top_txt   = sorted_candidates[0]
    top_score = sorted_scores[0]
    low_txt   = sorted_candidates[-1]
    low_score = sorted_scores[-1]

    ct_data = {
        "🔍 Clarity": (
            f"The query was: *\"{query}\"* — The model converted it into a "
            "high-dimensional vector and compared it with each candidate text using "
            "cosine similarity. The output is a score between 0 (no similarity) and 1 (identical meaning)."
        ),
        "✅ Accuracy": (
            "The model used is **all-MiniLM-L6-v2** from the `sentence-transformers` library "
            "(HuggingFace). It is a free, open-source model. No custom training or manual "
            "tokenization was performed."
        ),
        "📏 Precision": (
            f"The **highest** similarity score is **{top_score:.4f}** for the candidate: "
            f"*\"{top_txt}\"*. "
            f"The **lowest** score is **{low_score:.4f}** for: *\"{low_txt}\"*. "
            "All scores are reported to 4 decimal places."
        ),
        "🔗 Relevance": (
            "All three graphs (Bar Chart, Heatmap, PCA Plot) directly reflect the computed "
            "cosine similarity values. The bar chart ranks candidates, the heatmap shows all "
            "pairwise relationships, and the PCA plot shows semantic clustering in 2D space."
        ),
        "💡 Logic": (
            f"The top result (*\"{top_txt[:60]}…\"* if long) scores highest because its semantic "
            "meaning is closest to the query. Texts in the same domain share similar embedding "
            "directions, resulting in higher cosine similarity."
        ),
        "⭐ Significance": (
            f"The most important finding is that the top candidate achieved a score of "
            f"**{top_score:.4f}**, indicating strong semantic alignment with the query. "
            "This demonstrates the model's ability to capture meaning beyond exact word matching."
        ),
        "⚖️ Fairness (Limitation)": (
            "**Limitation:** The `all-MiniLM-L6-v2` model is trained primarily on English text. "
            "It may not perform well on multilingual inputs, highly technical jargon, or very "
            "short single-word inputs that lack context."
        ),
    }

    for standard, explanation in ct_data.items():
        with st.expander(standard, expanded=True):
            st.markdown(explanation)

    st.divider()
    st.caption(
        "Quiz Submission | NLP Lab | Shifa Tameer-e-Millat University | "
        "Model: all-MiniLM-L6-v2 | No preprocessing — No training"
    )
