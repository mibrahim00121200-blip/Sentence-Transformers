# 🧠 NLP Text Similarity Explorer

**Shifa Tameer-e-Millat University — NLP Lab Quiz**

## 📌 App Purpose
A Streamlit web app that uses a free pretrained NLP model to compute and visualize **text/sentence similarity** using cosine similarity on semantic embeddings.

## 🤖 Model Used
**`all-MiniLM-L6-v2`** from [Sentence Transformers (HuggingFace)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- Free & open-source
- No training performed
- No preprocessing performed

## 🚀 Streamlit App Link
👉 **[Click here to open the live app](<YOUR_STREAMLIT_LINK_HERE>)**

## 📊 Features
- Enter any query text and multiple candidate sentences
- Computes cosine similarity scores using the pretrained model
- **3 Graphs:**
  1. 📈 Bar Chart — ranked similarity scores
  2. 🌡️ Heatmap — pairwise similarity matrix
  3. 🗺️ 2D PCA Plot — semantic space visualization
- Paul's Critical Thinking Standards analysis panel

## 📁 Repository Structure
```
├── app.py            # Main Streamlit application
├── requirements.txt  # Required Python libraries
└── README.md         # This file
```

## ▶️ How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📸 Screenshots
*(Add screenshots of your running app here after deployment)*

## ⚠️ Rules Followed
- ✅ Free pretrained model only
- ✅ No manual preprocessing
- ✅ No model training
- ✅ No paid API
