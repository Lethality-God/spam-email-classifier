# 📩 Spam Message Classifier

A machine learning project to classify messages as **Spam** or **Ham** using multiple algorithms.

## 🚀 Models Used

* Logistic Regression
* Random Forest
* Multinomial Naive Bayes

## 📊 Results

### Logistic Regression

* Accuracy: **~100%**
* Confusion Matrix:

```
[[3618    6]
 [  10  645]]
```

### Random Forest

* Accuracy: **~100%**
* Confusion Matrix:

```
[[3623    1]
 [   5  650]]
```

### Multinomial Naive Bayes

* Accuracy: **~99%**
* Confusion Matrix:

```
[[3612   12]
 [  19  636]]
```

## 📁 Datasets

* Email Spam Detection Dataset – Sheikh Muhammad Abdullah (Kaggle)
* SMS Spam Collection Dataset – UCI ML (Kaggle)
* SMS Phishing Dataset – Fadli Fatih (Kaggle)

## ⚙️ Tech Stack

* Python
* Scikit-learn
* Pandas
* NumPy

## 🧠 Key Insight

Random Forest performed best with the lowest misclassification, while all models achieved near-perfect accuracy.

## 📌 Usage

```bash
git clone https://github.com/your-username/spam-classifier.git
cd spam-classifier
pip install -r requirements.txt
python main.py
```

---
