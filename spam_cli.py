import argparse
import joblib
import numpy as np
from scipy.sparse import hstack
import re


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


def predict_message(message, bundle, sender='UNKNOWN'):
    model = bundle["model"]
    vectorizer = bundle["vectorizer"]
    risk_max = bundle["risk_max"]
    sender_map = bundle["sender_map"]
    threshold = bundle["threshold"]

    message = message.lower()

    risk, flags = labelling(message)
    risk = risk / (risk_max + 1e-5)

    sender = sender.upper().strip()
    sender_score = sender_map.get(sender, 0)

    vec = vectorizer.transform([message])

    risk = np.array([[risk]])
    sender_score = np.array([[sender_score]])

    X = hstack([vec, risk, sender_score])

    prob = model.predict_proba(X)[0][1]
    label = 'spam' if prob > threshold else 'ham'

    print(f"\nPrediction: {label}")
    print(f"Confidence: {prob*100:.2f}%")
    print(f"Flags: {flags if flags else 'None'}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spam Detection CLI Tool")

    parser.add_argument("message", type=str, help="Message to classify")
    parser.add_argument("--sender", type=str, default="UNKNOWN",
                        help="Sender type (GOV, CONTACT, UNKNOWN)")

    args = parser.parse_args()

    bundle = joblib.load("model_bundle.pkl")

    predict_message(args.message, bundle, args.sender)
