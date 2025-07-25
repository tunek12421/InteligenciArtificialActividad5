import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Cargar el dataset de Titanic
df = pd.read_csv('../Ejercicio1/titanic_limpio.csv')
X = df.drop('Survived', axis=1)  # Corregido: era '2urvived'
y = df['Survived']  # Corregido: era '2urvived'

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ajustar un modelo de clasificación
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Interpretabilidad con SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualizar los valores SHAP
shap.summary_plot(shap_values, X_test)

# Interpretabilidad con LIME
import lime
import lime.lime_tabular

explainer_lime = lime.lime_tabular.LimeTabularExplainer(X_train.values,
feature_names=X_train.columns, class_names=['No', 'Sí'], mode='classification')

i = 0 # Índice del ejemplo a explicar
exp = explainer_lime.explain_instance(X_test.values[i], model.predict_proba)

# Alternativa a show_in_notebook()
print("Explicación LIME:")
print(exp.as_list())

# O guardar como HTML
exp.save_to_file('lime_explanation.html')
print("Explicación guardada en 'lime_explanation.html'")