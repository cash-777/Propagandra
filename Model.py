import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#load data set from a csv file
true_data = pd.read_csv('DataSet_Misinfo_TRUE.csv')
false_data = pd.read_csv('DataSet_Misinfo_FAKE.csv')

# Add labels
true_data['label'] = 1
false_data['label'] = 0

# Combine the datasets
data = pd.concat([true_data, false_data])

# Shuffle the data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

#removes missing data
data = data.dropna(subset=['text'])

X = data['text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#convert text dat to numerical format useing TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidif = vectorizer.fit_transform(X_train)
X_test_tfidif = vectorizer.transform(X_test)

# init Logistic Regression
model = LogisticRegression(max_iter=1000)

#train the model
model.fit(X_train_tfidif, y_train)

#make predictions
y_pred = model.predict(X_test_tfidif)

#see how well it cooked

print("accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))



import joblib

# Save the trained model
joblib.dump(model, 'logistic_regression_model.pkl')

# Save the TF-IDF vectorizer
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
