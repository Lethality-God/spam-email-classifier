import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split as TTS
from sklearn.linear_model import LogisticRegression as LR

df = pd.read_csv("mail_data.csv")

df.Message = df.Message.str.lower()
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=2)

X = vectorizer.fit_transform(df.Message)
y = df.Category

X_train, X_test, y_train, y_test = TTS(X,y,random_state=0)
model = LR(class_weight="balanced", C = 100, max_iter=1000)
model.fit(X_train, y_train)

predicted = model.predict(X_test)

tp,fp,tn,fn = 0,0,0,0

for i in range(0, len(y_test)):
    pre = predicted[i]
    act = y_test.iloc[i]

    if(pre=='spam' and act == 'spam'):
        tp += 1

    elif(pre=='spam' and act=='ham'):
        fp += 1

    elif(pre=='ham' and act == 'spam'):
        fn += 1

    else:
        tn+=1

print(f"True Positive = {tp} \n True Negative = {tn} \n False Positive = {fp} \n False Negative = {fn}")
print(f"Accuracy: {(tp+tn)/(tp+tn+fp+fn)*100}")
print(f"Recall: {tp/(tp+fn)*100}")
print(f"Precision: {tp/(tp+fp)*100}")
