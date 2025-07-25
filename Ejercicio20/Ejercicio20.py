import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
# Cargar el dataset de 20 Newsgroups
newsgroups = fetch_20newsgroups(subset='all', categories=['alt.atheism', 
'soc.religion.christian'], remove=('headers', 'footers', 'quotes'))
X = newsgroups.data
y = newsgroups.target
# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Vectorizar el texto

vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)

X_test_vec = vectorizer.transform(X_test)
# Ajustar un modelo de clasificación
model = OneVsRestClassifier(LogisticRegression())
model.fit(X_train_vec, y_train)
# Predecir y evaluar
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred, target_names=newsgroups.target_names))

