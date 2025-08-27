# fake-news-ml-project

# 📰 Fake News Detection with Logistic Regression

This project detects **fake vs real news articles** using **Natural Language Processing (NLP)** and **Machine Learning**.
It uses the **Fake News Challenge dataset (FakeNewsKDD2020)**, processes text data, and applies **Logistic Regression** to classify news.

---

## 🚀 Features

* Load dataset directly from **Kaggle competitions** in **Google Colab**
* Clean and preprocess text (remove stopwords, stemming with PorterStemmer)
* Convert text to numerical features with **TF-IDF Vectorization**
* Train/test split and Logistic Regression model training
* Evaluate with **accuracy score**
* Test with custom samples to check predictions

---

## 📂 Dataset

* **Competition:** [FakeNewsKDD2020](https://www.kaggle.com/competitions/fakenewskdd2020)
* **Files:** `train.csv`, `test.csv`

Each news item has:

* **text**: The news article content
* **label**: 0 = Real, 1 = Fake

---

## ⚙️ Installation & Setup

### 1. Upload Kaggle API key

```python
from google.colab import files
files.upload()   # upload kaggle.json
```

### 2. Configure Kaggle

```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

### 3. Download Dataset

```bash
!kaggle competitions download -c fakenewskdd2020
```

### 4. Extract Files

```python
from zipfile import ZipFile

with ZipFile('/content/fakenewskdd2020.zip', 'r') as z:
    z.extractall('/content/')
```

---

## 🧹 Data Preprocessing

* Remove special characters & digits
* Lowercase text
* Remove **stopwords**
* Apply **PorterStemmer** for stemming

```python
def stemming(text):
    stemmed_text = re.sub('[^a-zA-Z]', ' ', text)
    stemmed_text = stemmed_text.lower().split()
    stemmed_text = [port_stem.stem(word) for word in stemmed_text if word not in stopwords.words('english')]
    return ' '.join(stemmed_text)

news_dataset['text'] = news_dataset['text'].apply(stemming)
```

---

## 🔎 Feature Extraction

Convert text into **TF-IDF vectors**:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)
```

---

## 🏋️ Model Training

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, Y_train)
```

---

## 📊 Evaluation

```python
from sklearn.metrics import accuracy_score

# Training accuracy
train_pred = model.predict(X_train)
print("Training Accuracy:", accuracy_score(train_pred, Y_train))

# Testing accuracy
test_pred = model.predict(X_test)
print("Testing Accuracy:", accuracy_score(test_pred, Y_test))
```

---

## 🔮 Prediction Example

```python
X_new = X_test[503].reshape(1, -1)
prediction = model.predict(X_new)

if int(prediction[0]) == 0:
    print("The news is Real")
else:
    print("The news is Fake")

print("Actual Label:", Y_test[503])
```

---

## 📌 Results

* Logistic Regression achieved:

  * **Training Accuracy:** \~99%
  * **Testing Accuracy:** \~95–97%

---

## 📖 Future Improvements

* Try advanced models: **XGBoost, RandomForest, LSTM**
* Add explainability (SHAP/LIME)
* Deploy with **Flask/Streamlit**

---

✨ Built with **Python, Scikit-learn, NLTK, TF-IDF, Logistic Regression**

---
