# 💬 Toxic Comment Classification using NLP, TF-IDF, and Logistic Regression

This project focuses on detecting multiple types of toxicity in user comments using traditional Natural Language Processing (NLP) techniques and a lightweight logistic regression model. It handles **multi-label classification**, meaning a single comment can belong to multiple toxic categories simultaneously. This model is suitable for real-time content moderation and as a baseline for more complex deep learning solutions.

---

## 🧠 Problem Statement

Online platforms often struggle with moderating toxic or harmful comments that include insults, threats, or hate speech. The objective here is to build a model that can classify text into one or more of the following categories:

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

---

## 📊 Dataset

The dataset is derived from the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge), where each row contains a text comment and one or more binary labels indicating types of toxicity.

---

## 🔧 Preprocessing Steps

- Removed non-alphabetical characters using regex
- Converted text to lowercase
- Tokenized text and removed stopwords (except "not")
- Applied stemming using NLTK’s Porter Stemmer
- Handled edge cases like recursion errors during stemming
- Built a clean text corpus for vectorization

---

## 🔢 Feature Extraction

- Used `TfidfVectorizer` with:
  - `max_features=20000`
  - `ngram_range=(1, 2)` for capturing unigrams and bigrams
- Transformed cleaned corpus into sparse matrix format for model input

---

## 🧪 Model Training

- Split data into 80% training and 20% test sets
- Used `OneVsRestClassifier` strategy to support multi-label classification
- Base classifier: `LogisticRegression` (solver = `'liblinear'` for efficiency)

---

## 📈 Evaluation

- Evaluated the model using **ROC AUC score** for each individual label
- Computed **macro-averaged ROC AUC** to get an overall performance metric
- ROC AUC was chosen due to class imbalance and multi-label nature

---

## 🛠️ Tech Stack

- Python 3
- NLTK
- Scikit-learn
- NumPy / Pandas

---

## 🚀 Future Improvements

- Try advanced models like BERT or DistilBERT for contextual understanding
- Handle class imbalance with sample weighting or focal loss
- Apply threshold tuning to improve precision/recall per label

---

## 📎 Credits

- Dataset: [Kaggle Toxic Comment Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- Tools: Scikit-learn, NLTK, Python

---

## 🧾 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
