# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 16:23:09 2020

@author: Murilo

Algoritmo kNN (k-Nearest Neighbour).
     
   Precisão:
     - Todos os tratamentos: 0,8305
     - Sem StandardScaler: 0,7692
"""

import pandas as pd

# criação da base de dados
base = pd.read_csv('census.csv')

# previsores e classe
previsores = base.iloc[:, :14].values
classe = base.iloc[:, 14].values

# transformação de variáveis categóricas (não numéricas)
# colunas 1, 3, 5, 6, 7, 8, 9, 13 e 14(classe)
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
classe = labelEncoder.fit_transform(classe)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
oneHotEncorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1,3,5,6,7,8,9,13])], remainder='passthrough')
previsores = oneHotEncorder.fit_transform(previsores).toarray()

# padronizando os valores dos previsores
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores[:, 102:] = scaler.fit_transform(previsores[:, 102:])

# separação entre base de treinamento e testes
from sklearn.model_selection import train_test_split
previsoresTreinamento, previsoresTeste, classeTreinamento, classeTeste = train_test_split(previsores, classe, random_state=0)

# treinamento
from sklearn.neighbors import KNeighborsClassifier
classificador = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=3)
classificador.fit(previsoresTreinamento, classeTreinamento)
resultado = classificador.predict(previsoresTeste)

# análise dos resultados
from sklearn.metrics import confusion_matrix, accuracy_score
matriz = confusion_matrix(classeTeste, resultado)
precisao = accuracy_score(classeTeste, resultado)
