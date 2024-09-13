import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
filepath = "data/short_clean_v2.csv"
df = pd.read_csv(filepath, encoding='cp1252')

x = df["Text"]
y = df["Label"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

pipSVC = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.9)),
    ('clf', LinearSVC())
])

pipSVC.fit(x_train, y_train)

predictSVC = pipSVC.predict(x_test)

scoreSVC = accuracy_score(y_test, predictSVC)
print(f'Accuracy: {scoreSVC}')
print(classification_report(y_test, predictSVC))

joblib.dump(pipSVC, 'svc_pipeline_model.pkl')
