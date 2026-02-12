# NLP_Assignment_1

**Team Members:** Maria Goicoechea, Joaquin Orradre, Paula Pina

## Description

This repository contains the code and analysis for NLP Assignment 1, focusing on establishing robust baseline models for text classification using the QEvasion dataset. The assignment follows the "Baselines Before Breakthroughs" philosophy, comparing sparse (TF-IDF, Count Vectors) and dense (Word2Vec, GloVe, FastText) feature representations.

## Repository Structure

```
├── report.typ              # Comprehensive technical report (Typst format)
├── report.pdf              # Compiled PDF report
├── Baseline.ipynb          # Sparse vs. dense feature comparison
├── Phase_1.ipynb           # Preprocessing ablation studies
├── Phase_2.ipynb           # Feature engineering experiments
├── EDA.ipynb              # Exploratory data analysis
├── expert_vote.ipynb      # Annotation resolution methodology
├── src/
│   ├── preprocessing.py   # Data loading and splitting utilities
│   └── evaluate.py        # Evaluation metrics and visualization
└── README.md
```

## Dataset

**QEvasion** (ailsntua/QEvasion) - Presidential interview Q&A pairs
- Training set: 3,448 samples
- Test set: 308 samples
- Tasks: Multi-label classification (clarity and evasion labels)

## Key Results

- **Best Model:** CountVectorizer + Logistic Regression (F1-Macro: 0.495)
- **Sparse vs Dense:** Sparse features outperform dense embeddings by 27%
- **N-grams:** Unigrams perform best for this dataset size
- **Preprocessing:** Character n-grams and vocabulary size matter most

## Report Compilation

The technical report is written in Typst format. To compile:

### Option 1: Using Typst CLI
```bash
# Install Typst (if not already installed)
curl -fsSL https://github.com/typst/typst/releases/download/v0.11.1/typst-x86_64-unknown-linux-musl.tar.xz | tar -xJ

# Compile report
./typst-x86_64-unknown-linux-musl/typst compile report.typ report.pdf
```

### Option 2: Using Typst Web App
1. Visit https://typst.app/
2. Upload `report.typ`
3. Export as PDF

## Requirements

```bash
pip install numpy pandas scikit-learn datasets gensim jupyter
```

## Reproducibility

All experiments use `random_state=42` for reproducibility. The complete methodology follows strict academic standards with stratified splitting and 5-fold cross-validation.

## License

See LICENSE file for details.