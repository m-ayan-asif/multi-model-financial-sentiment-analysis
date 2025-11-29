# multi-model-financial-sentiment-analysis
Benchmarking multiple sentiment analysis approaches for financial domain text i.e FinBERT, zero-shot LLMs, and a custom RAG pipeline, paired with LDA topic modeling and extensive evaluation. Designed for reproducible deep learning workflows.




Assignment 3 — Topic Modeling and Sentiment Analysis in Financial Text
Author: Ayan Asif
Course: Deep Learning for Perception (CS4045)
Instructor: Dr. Ahmad Raza Shahid

============================================================
1. Overview
============================================================
This repository contains the implementation for Assignment #3.  
It includes:
- Data preprocessing and EDA
- LDA topic modeling (k = 35, with coherence experiments)
- FinBERT-based sentiment analysis
- Local LLM-based sentiment analysis (two baselines)
- RAG-enhanced sentiment analysis using FAISS + embeddings
- Fine-tuning decision (FinBERT accuracy > 90% → fine-tuning not required)
- All evaluation metrics and confusion matrices
- Reproducible artifacts (embeddings, FAISS index, CSVs)

The entire workflow is reproducible by running the notebook
`assignment3_notebook.ipynb` from top to bottom.

============================================================
2. Environment Setup
============================================================

(1) Create Conda environment
--------------------------------
conda create -n fintext python=3.10
conda activate fintext

(2) Install dependencies
--------------------------------
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers sentence-transformers faiss-cpu gensim nltk scikit-learn pandas matplotlib seaborn

Optional:
pip install jupyterlab ipywidgets

(3) Enable GPU (if available)
--------------------------------
The notebook auto-detects CUDA and runs FinBERT + LLM inference on GPU.

============================================================
3. Running the Notebook
============================================================

Open:
    assignment3_notebook.ipynb

Then run the notebook top-to-bottom.

The notebook will automatically:

- Load and preprocess the dataset
- Run LDA and compute topic coherence
- Run FinBERT inference (batching on GPU)
- Run two Local LLM zero-shot classifiers
- Build embeddings using all-MiniLM-L6-v2
- Build and query a FAISS index for RAG
- Evaluate all systems
- Export final predictions to resultsWithAllModels.csv

============================================================
4. Artifacts Included
============================================================

The submission includes:

- assignment3_notebook.ipynb
- resultsWithAllModels.csv
- embs.npy
- faiss_index.idx
- cm_finbert.png
- cm_llm1.png
- cm_llm2.png
- cm_rag.png
- assignment3_report.pdf (compiled LaTeX report)

============================================================
5. Reproducing Results
============================================================

After installing dependencies:

Run the notebook from top to bottom.  
Random seeds are set globally (SEED=42) to ensure reproducibility.

FinBERT should reproduce:
    Accuracy ≈ 0.9236

Local LLMs should produce:
    ≈ 0.68  (DeBERTa NLI)
    ≈ 0.34  (DistilBERT MNLI)

RAG should produce:
    ≈ 0.18, 0.16, 0.17 for k = 5, 3, 10 respectively

============================================================
6. Notes
============================================================
FinBERT achieved >90% accuracy, so fine-tuning was NOT required based on the assignment's fine-tuning rule.

All embeddings, indexes, and result CSVs are saved in this directory and can be reused without re-running the full pipeline.
