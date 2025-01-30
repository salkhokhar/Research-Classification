# Medical Abstract Classification Using NLP and Machine Learning

## Overview

This project classifies medical research abstracts from PubMed on diseases like Diabetes, Alzheimer's, Leukemia, Stroke, and HIV. It includes preprocessing, feature extraction, visualization, and classification using multiple machine learning models, including Naive Bayes, KNN, Random Forest and BERT.

---

## Table of Contents

- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Feature Extraction](#feature-extraction)
- [Machine Learning Models](#machine-learning-models)
- [BERT Model](#bert-model)
- [Performance Evaluation](#performance-evaluation)
- [Visualization](#visualization)
- [Model Performance Comparison](#model-performance-comparison)
- [Usage](#usage)

---

## Dataset

The datasets are sourced from **Hugging Face Datasets**, containing PubMed abstracts related to:
- **Diabetes**
- **Alzheimer's Disease**
- **Leukemia**
- **Stroke**
- **HIV/AIDS**

Each dataset is loaded and converted into a Pandas DataFrame for further processing.

---

## Preprocessing

The preprocessing pipeline includes:
1. **Text Cleaning**: Removing punctuation, converting to lowercase, stripping spaces.
2. **Stopword Removal**: Using `nltk` stopwords and domain-specific stopwords.
3. **Tokenization**: Splitting text into words.
4. **Lemmatization**: Converting words to their base forms.
5. **Sampling**: Taking 200 random abstracts per category for balance.

---

## Feature Extraction

Different feature extraction methods are applied:
- **Bag of Words (BoW)**
- **TF-IDF (Term Frequency-Inverse Document Frequency)**
- **N-Grams (Bigrams, Trigrams)**

Scikit-learn's `CountVectorizer` and `TfidfVectorizer` are used for feature engineering.

---

## Machine Learning Models

The following models are trained and evaluated:
- **Naive Bayes** (MultinomialNB)
- **K-Nearest Neighbors (KNN)**
- **Random Forest**
- **BERT**

Each model undergoes **10-fold cross-validation** to evaluate performance.

---

## BERT Model

A fine-tuned **BERT (Bidirectional Encoder Representations from Transformers)** model is used for classification:
- **DistilBERT** is fine-tuned with training and validation datasets.
- **Custom Tokenization** is performed using `BertTokenizer`.
- **K-Fold Cross-Validation** ensures robust evaluation.

---

## Performance Evaluation

Evaluation metrics used:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**
- **ROC Curves**

Performance metrics are stored in `model_performance_metrics.csv`.

---

## Visualization

### 1. Word Frequency Analysis
- **Unigrams** and **Bigrams** are plotted to show the most frequent words.

### 2. Word Cloud
- A **word cloud** visualizes the most common words across all abstracts.

### 3. Confusion Matrices
- **Heatmaps** display model performance for each classification method.

### 4. Prediction Analysis
- **Comparison of correct and incorrect predictions** for each disease label.

---

## Model Performance Comparison

| Model                          | Mean Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score |
|--------------------------------|--------------|--------------------|----------------|------------------|
| Naive Bayes (BOW)             | 0.8025       | 0.8027             | 0.8025         | 0.8023           |
| Naive Bayes (TF-IDF)          | 0.79375      | 0.8068             | 0.79375        | 0.7952           |
| Naive Bayes (n-gram)          | 0.665        | 0.7710             | 0.665          | 0.6659           |
| K-Nearest Neighbors (BOW)     | 0.35         | 0.4428             | 0.35           | 0.3377           |
| K-Nearest Neighbors (TF-IDF)  | 0.7125       | 0.7228             | 0.7125         | 0.7089           |
| K-Nearest Neighbors (n-gram)  | 0.18625      | 0.1321             | 0.18625        | 0.1236           |
| Random Forest (BOW)           | 0.845        | 0.8500             | 0.845          | 0.8428           |
| Random Forest (TF-IDF)        | 0.82625      | 0.8381             | 0.82625        | 0.8253           |
| Random Forest (n-gram)        | 0.65125      | 0.7438             | 0.65125        | 0.6511           |
| BERT_with_stop_words          | 0.85         | 0.8499             | 0.8489         | 0.8447           |
| BERT_without_stop_words       | 0.8675       | 0.8663             | 0.8687         | 0.8637           |



## Usage

### 1. Clone the Repository

First, download the repository to your local machine:

```sh
git clone https://github.com/salkhokhar/Research-Classification.git
cd Research-Classification
```

### 2. Install Dependencies
Ensure you have all the required libraries installed:

```sh
pip install -r requirements.txt
```

### 3. Run the Data Processing Script
Execute the preprocessing and feature extraction script:

```sh
python research_classifier.py
```

