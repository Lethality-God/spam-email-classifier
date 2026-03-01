import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split as TTS
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.metrics import classification_report, confusion_matrix
import re
from scipy.sparse import hstack
import joblib

def labelling(message):

        url_pattern = r'(https?://\S+|www\.\S+)'
        x = re.findall(url_pattern, message)
        
        risk = 0
        flags = []

        for url in x:
            if url.startswith("http://"):
                risk += 1
                flags.append("No HTTPS")

            if any(short in url for short in ["bit.ly", "tinyurl", "goo.gl"]):
                risk += 2
                flags.append("Shortened URL")

            if any(word in url.lower() for word in ["login", "verify", "bank", "update"]):
                risk += 1
                flags.append("Suspicious keyword")

        return risk, ", ".join(flags)

try:
    bundle = joblib.load("model_bundle.pkl")

    LR_model = bundle["LR_model"]
    RF_model = bundle["RF_model"]
    MNB_model = bundle["MNB_model"]
    vectorizer = bundle["vectorizer"]
    risk_max = bundle["risk_max"]
    sender_map = bundle["sender_map"]
    threshold = bundle["threshold"]

except FileNotFoundError:
    threshold = 0.6
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df2 = pd.read_csv("phishing.csv")
    df3 = pd.read_csv("mail_data.csv")

    df2 = df2.drop(columns=["URL", "EMAIL", "PHONE"])

    df['v1'] = df['v1'].str.lower()

    df2 = df2.rename(columns={
        "LABEL": "v1",
        "TEXT": "v2"
    })
    df3 = df3.rename(columns={
        "Category": "v1",
        "Message": "v2"
    })

    df2['v2'] = df2['v2'].str.lower()
    df3['v2'] = df3['v2'].str.lower()


    df = df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])
    df=pd.concat([df, df2], ignore_index=True)
    df=pd.concat([df, df3], ignore_index=True)
    df['url_risk'] = 0
    df['flags'] = 'NONE'
    df['sender'] = 'UNKNOWN'

    df.loc[(df['v1'] == 'ham') & (df['v2'].str.contains('bank', case=False)),'sender'] = 'GOV'

    ham_indices = df[df['v1'] == 'ham'].sample(frac=0.3, random_state=42).index

    spam_indices = df[df['v1'] == 'spam'].sample(frac=0.3, random_state=42).index

    df.loc[ham_indices, 'sender'] = 'CONTACT'
    df.loc[spam_indices, 'sender'] = 'CONTACT'
    df['sender'] = df['sender'].str.upper().str.strip()

    sender_map = {
        'GOV': 1,
        'CONTACT': 0.5,
        'UNKNOWN': 0
    }

    df['sender_score'] = df['sender'].map(sender_map)


    df.v2 = df.v2.str.lower()
    df['v1'] = df['v1'].replace({
        'spam': 'spam',
        'ham': 'ham',
        'Smishing': 'spam',
        'smishing': 'spam',
        'Spam': 'spam'
    })

    df[['url_risk', 'flags']] = df['v2'].apply(lambda x: pd.Series(labelling(x)))

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=2)

    y = df.v1
    X_train_text, X_test_text, y_train, y_test, \
    X_train_risk, X_test_risk, \
    X_train_sender, X_test_sender = TTS(
        df['v2'],
        y,
        df['url_risk'],
        df['sender_score'],
        random_state=0,
        stratify=y
    )
    print(df.head())

    X_train_text = vectorizer.fit_transform(X_train_text)
    X_test_text = vectorizer.transform(X_test_text)

    X_train_risk = np.array(X_train_risk).reshape(-1, 1)
    X_test_risk = np.array(X_test_risk).reshape(-1, 1)
    X_train_risk = X_train_risk / (X_train_risk.max() + 1e-5)
    X_test_risk = X_test_risk / (X_train_risk.max() + 1e-5)

    X_train_sender = np.array(X_train_sender).reshape(-1, 1)
    X_test_sender = np.array(X_test_sender).reshape(-1, 1)


    X_train = hstack([X_train_text, X_train_risk, X_train_sender])
    X_test = hstack([X_test_text, X_test_risk, X_test_sender])



    def LR_model_generator(X_train, X_test, y_train, y_test, threshold):

        LR_model = LR(class_weight="balanced", C = 5, max_iter=1000)
        LR_model.fit(X_train, y_train)

        spam_prob = LR_model.predict_proba(X_test)[:, 1]
        predicted = np.where(spam_prob >= threshold, 'spam', 'ham')

        print(classification_report(y_test, predicted))

        print(confusion_matrix(y_test, predicted))
        return LR_model
    
    def RF_model_generator(X_train, X_test, y_train, y_test, threshold):

        RF_model = RFC(
            class_weight="balanced",
            random_state=0)
        
        RF_model.fit(X_train, y_train)

        spam_prob = RF_model.predict_proba(X_test)[:, 1]
        predicted = np.where(spam_prob >= threshold, 'spam', 'ham')

        print(classification_report(y_test, predicted))

        print(confusion_matrix(y_test, predicted))
        return RF_model
    
    def MNB_model_generator(X_train, X_test, y_train, y_test, threshold):

        MNB_model = MNB()
        
        MNB_model.fit(X_train, y_train)

        spam_prob = MNB_model.predict_proba(X_test)[:, 1]
        predicted = np.where(spam_prob >= threshold, 'spam', 'ham')

        print(classification_report(y_test, predicted))

        print(confusion_matrix(y_test, predicted))
        return MNB_model
    print("For same threshold:")
    print("Logistic Regression:")
    LR_model = LR_model_generator(X_train, X_test, y_train, y_test, threshold=threshold)
    print("Random Forest:")
    RF_model = RF_model_generator(X_train, X_test, y_train, y_test, threshold=threshold)
    print("MNB:")
    MNB_model = MNB_model_generator(X_train, X_test, y_train, y_test, threshold=threshold)

    print("For local best threshold:")
    print("Logistic Regression:")
    LR_model = LR_model_generator(X_train, X_test, y_train, y_test, threshold=0.6)
    print("Random Forest:")
    RF_model = RF_model_generator(X_train, X_test, y_train, y_test, threshold=0.2)
    print("MNB:")
    MNB_model = MNB_model_generator(X_train, X_test, y_train, y_test, threshold=0.2)


    risk_max = X_train_risk.max()

    bundle = {
        "LR_model": LR_model,
        "RF_model": RF_model,
        "MNB_model": MNB_model,
        "vectorizer": vectorizer,
        "risk_max": risk_max,
        "sender_map": sender_map,
        "threshold": 0.6,
    }

    joblib.dump(bundle, "model_bundle.pkl")

def predict_message(message, LR_model, RF_model, MNB_model, vectorizer, risk_max, sender_map, threshold,sender='UNKNOWN'):
    message = message.lower()

    if sender.upper().strip() == "GOV":
        print("Prediction: ham")
        print("Confidence: 100.00%")
        print("Flags: Trusted GOV sender override")
        return
    
    risk, flags = labelling(message)
    risk = risk / (risk_max + 1e-5)
    
    sender = sender.upper().strip()
    sender_score = sender_map.get(sender, 0)
    
    vec = vectorizer.transform([message])
    
    risk = np.array([[risk]])
    sender_score = np.array([[sender_score]])
    
    X = hstack([vec, risk, sender_score])
    
    prob1 = LR_model.predict_proba(X)[0][1]
    prob2 = RF_model.predict_proba(X)[0][1]
    prob3 = MNB_model.predict_proba(X)[0][1]
    prob = (prob1+prob2+prob3)/3
    label = 'spam' if prob > threshold else 'ham'
    
    print(f"Prediction: {label}")
    print(f"Confidence_LR: {prob1*100:.2f}%")
    print(f"Confidence_RF: {prob2*100:.2f}%")
    print(f"Confidence_MNB: {prob3*100:.2f}%")
    print(f"Confidence: {prob*100:.2f}%")
    print(f"Flags: {flags if flags else 'None'}")
