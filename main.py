import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split as TTS
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("mail_data.csv")

df.Message = df.Message.str.lower()
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=2)

y = df.Category
X_train, X_test, y_train, y_test = TTS(df.Message, y, random_state=0)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


def model_generator(X_train, X_test, y_train, y_test, threshold):

    model = LR(class_weight="balanced", C = 100, max_iter=1000)
    model.fit(X_train, y_train)

    spam_prob = model.predict_proba(X_test)[:, 1]
    predicted = np.where(spam_prob >= threshold, 'spam', 'ham')

    print(f"Report for thereshold {threshold}:")
    print(classification_report(y_test, predicted))

    print(confusion_matrix(y_test, predicted))

model_generator(X_train, X_test, y_train, y_test, 0.6)

