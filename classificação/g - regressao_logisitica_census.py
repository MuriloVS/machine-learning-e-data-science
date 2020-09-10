# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 22:06:11 2020

@author: Murilo

Algoritmo de Regressão Logística.

Precisão:
  - Todos os tratamentos, max_iter=100: 0,8505
  - Todos os tratamentos, max_iter=200: 0,8508
  - Todos os tratamentos, max_iter=400: 0,8508
  - Sem StandardScaler, max_iter=200: 0,7949

"""

import pandas as pd

# criação da base de dados
base = pd.read_csv('census.csv')

# previsores e classe
previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values
                
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# transformação de variáveis categóricas (não numéricas)
# colunas 1, 3, 5, 6, 7, 8, 9, 13 e 14 (classe)
columns = []
for i in range(len(previsores[0])):
    if type(previsores[:, i][0]) == type('str'):
        columns.append(i)
column_tranformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), columns)],remainder='passthrough')
previsores = column_tranformer.fit_transform(previsores).toarray()

labelencorder_classe = LabelEncoder()
classe = labelencorder_classe.fit_transform(classe)

# padronizando os valores dos previsores
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores[:, 102:] = scaler.fit_transform(previsores[:, 102:])

# separação entre base de treinamento e testes
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, random_state=0)

# treinamento
from sklearn.linear_model import LogisticRegression
classificador = LogisticRegression(solver='lbfgs', max_iter=400)
classificador.fit(previsores_treinamento, classe_treinamento)
resultado = classificador.predict(previsores_teste)
# print(classificador.intercept_)
# print(classificador.coef_)

# análise dos resultados
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, resultado)
matriz = confusion_matrix(classe_teste, resultado)

import collections
collections.Counter(classe_teste)