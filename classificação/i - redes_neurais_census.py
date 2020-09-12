# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 18:55:09 2020

@author: Murilo

Precisão:
  - Todos os tratamentos (sem alterar parâmetros): 0,844
  - Todos os tratamento (max_iter=1000, tol=0.000001): 0,834
  
  - Todos os tratamento (max_iter=1000, tol=0.000001, activation="tanh"): 0,994
  
  - Todos os tratamento (max_iter=1000, tol=0.000001, activation="logistic"): 0,992
  
  - Todos os tratamento (max_iter=1000, tol=0.000001, solver="lbfgs"): 0,
  
  - Labelencoder + escalonamento (sem alterar parâmetros): 0,848
  - Labelencoder + escalonamento (max_iter=1000, tol=0.000001): 0,847

  
  - Sem escalonamento (sem alterar parâmetros): 0,794
"""

import pandas as pd

# criação da base de dados
base = pd.read_csv('census.csv')

# previsores e classe
previsores = base.iloc[:, :14].values
classe = base.iloc[:, 14].values

# transformação de variáveis categóricas (não numéricas)
# colunas 1, 3, 5, 6, 7, 8, 9, 13 e 14 (classe)
'''from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
columns = []
for i in range(len(previsores[0])):
    if type(previsores[:, i][0]) == type('str'):
        columns.append(i)
column_tranformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), columns)],remainder='passthrough')
previsores = column_tranformer.fit_transform(previsores).toarray()'''

from sklearn.preprocessing import LabelEncoder
labelEncoder_previsores = LabelEncoder()
for i in range(len(previsores[0])):
    if type(previsores[:, i][0]) == type('str'):
        previsores[:,i] = labelEncoder_previsores.fit_transform(previsores[:,i])

# padronizando os valores dos previsores
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# separação entre base de treinamento e testes
from sklearn.model_selection import train_test_split
previsoresTreinamento, previsoresTeste, classeTreinamento, classeTeste = train_test_split(previsores,
                                                                                          classe,
                                                                                          random_state=0)

# treinamento
from sklearn.neural_network import MLPClassifier
classificador = MLPClassifier(max_iter=1000, tol=0.000001)
classificador.fit(previsoresTreinamento, classeTreinamento)
resultado = classificador.predict(previsoresTeste)

# análise dos resultados
from sklearn.metrics import confusion_matrix, accuracy_score
matriz = confusion_matrix(classeTeste, resultado)
precisao = accuracy_score(classeTeste, resultado)
