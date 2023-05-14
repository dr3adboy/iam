# CREACION DE MODELO
# Importacion de librerias

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("/content/heart.csv")

# Escalamiento
from sklearn.preprocessing import RobustScaler

# Train Test Split
from sklearn.model_selection import train_test_split

# Modelos
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Metricas
from sklearn.metrics import accuracy_score, classification_report, roc_curve

# Cross Validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

print('Packages imported...')

# Creación de una copia del df
df1 = df

# Seleccion de las columnas a ser escaladas y codificadas
cat_cols = ['sex','exng','caa','cp','fbs','restecg','slp','thall']
con_cols = ["age","trtbps","chol","thalachh","oldpeak"]

# Codificando las columnas categoricas
df1 = pd.get_dummies(df1, columns = cat_cols, drop_first = True)

# Definiendo los atributos independientes y el atributo dependiente
X = df1.drop(['output'],axis=1)
y = df1[['output']]

# Instanciando el escalador
scaler = RobustScaler()

# Escalando los atributos continuos
X[con_cols] = scaler.fit_transform(X[con_cols])
print("Las primeras 5 filas de X")
X.head()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
print("El tamaño de X_train es      ", X_train.shape)
print("El tamaño de X_test es        ",X_test.shape)
print("El tamaño de y_train es       ",y_train.shape)
print("El tamaño de y_test es        ",y_test.shape)

# Instanciamiento del objeto
logreg = LogisticRegression()

# Ajustando el objeto
logreg.fit(X_train, y_train)

# Calculo de probabilidades
y_pred_proba = logreg.predict_proba(X_test)

# Encontrando los valores predichos
y_pred = np.argmax(y_pred_proba,axis=1)

# Impresión de la prueba de precisión
print("El puntaje de precisión en la prueba de Regresión Logística es ", accuracy_score(y_test, y_pred))
