# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 15:55:45 2020

@author: Murilo

Precisão:
  - Todos os tratamentos (linear): 0,851
  - Sem StandardScaler (linear): 0,791
  
  - Todos os tratamentos (rbf): 0,855
  - Todos os tratamentos (rbf, C=0.9): 0,855
  - Todos os tratamentos (rbf, C=2): 0,855 
  - Sem StandardScaler (rbf): ?
  
  - Todos os tratamentos (poly): 0,850
  - Sem StandardScaler (poly): ?
  
  - Todos os tratamentos (sigmoid): 0,774
  - Sem StandardScaler (sigmoid): ?
"""

import pandas as pd

# criação da base de dados
base = pd.read_csv('census.csv')

# previsores e classe
previsores = base.iloc[:, :14].values
classe = base.iloc[:, 14].values

# transformação de variáveis categóricas (não numéricas)
# colunas 1, 3, 5, 6, 7, 8, 9, 13 e 14 (classe)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
columns = []
for i in range(len(previsores[0])):
    if type(previsores[:, i][0]) == type('str'):
        columns.append(i)
column_tranformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), columns)],remainder='passthrough')
previsores = column_tranformer.fit_transform(previsores).toarray()

# padronizando os valores dos previsores
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores[:, 102:] = scaler.fit_transform(previsores[:, 102:])

# separação entre base de treinamento e testes
from sklearn.model_selection import train_test_split
previsoresTreinamento, previsoresTeste, classeTreinamento, classeTeste = train_test_split(previsores, classe, random_state=0)

# treinamento
from sklearn.svm import SVC
classificador = SVC(kernel='rbf', random_state=1, C=2)
classificador.fit(previsoresTreinamento, classeTreinamento)
resultado = classificador.predict(previsoresTeste)

# análise dos resultados
from sklearn.metrics import confusion_matrix, accuracy_score
matriz = confusion_matrix(classeTeste, resultado)
precisao = accuracy_score(classeTeste, resultado)
