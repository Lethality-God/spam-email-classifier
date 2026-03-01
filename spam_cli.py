import argparse
import joblib
from classifier import predict_message


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("message",type=str)

    parser.add_argument("--sender",type=str,default="UNKNOWN",)
    args = parser.parse_args()

    try:
        bundle = joblib.load("model_bundle.pkl")
    except FileNotFoundError:
        print("Model not found")
        return
    LR_model = bundle["LR_model"]
    RF_model = bundle["RF_model"]
    MNB_model = bundle["MNB_model"]
    vectorizer = bundle["vectorizer"]
    risk_max = bundle["risk_max"]
    sender_map = bundle["sender_map"]
    threshold = bundle["threshold"]
    predict_message(message=args.message,LR_model=LR_model,RF_model=RF_model, MNB_model=MNB_model,vectorizer=vectorizer, risk_max=risk_max,sender_map=sender_map, threshold=threshold, sender=args.sender)


if __name__ == "__main__":
    main()