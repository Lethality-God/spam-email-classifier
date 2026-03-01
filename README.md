Spam Message classifier bulit with Logistic Regression, Random Forest and Multinomial Naive Bayes.

Current stats:

Logistic Regression:
              precision    recall  f1-score   support

         ham       1.00      1.00      1.00      3624
        spam       0.99      0.98      0.99       655

    accuracy                           1.00      4279
   macro avg       0.99      0.99      0.99      4279
weighted avg       1.00      1.00      1.00      4279

[[3618    6]
 [  10  645]]
Random Forest:
              precision    recall  f1-score   support

         ham       1.00      1.00      1.00      3624
        spam       1.00      0.99      1.00       655

    accuracy                           1.00      4279
   macro avg       1.00      1.00      1.00      4279
weighted avg       1.00      1.00      1.00      4279

[[3623    1]
 [   5  650]]
MNB:
              precision    recall  f1-score   support

         ham       0.99      1.00      1.00      3624
        spam       0.98      0.97      0.98       655

    accuracy                           0.99      4279
   macro avg       0.99      0.98      0.99      4279
weighted avg       0.99      0.99      0.99      4279

[[3612   12]
 [  19  636]]

Datasets by Sheikh Muhammad Abdullah (https://www.kaggle.com/datasets/abdmental01/email-spam-dedection/data), UCI Machine Learning (https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) and Fadli Fatih (https://www.kaggle.com/datasets/fadlifatih/sms-phishing-dataset) from Kaggle.

