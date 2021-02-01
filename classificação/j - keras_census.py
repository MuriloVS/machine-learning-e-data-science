# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 23:14:00 2020

@author: mvs_r

Precisão (os dados variam - melhor resultado anotado):
  - Todos os tratamentos (batch_size=10, epochs=100): 0,8129
  - Sem onehotencoder (batch_size=10, epochs=100): 0,8501
  
  - Sem escalonamento (batch_size=10, epochs=100): ?
"""

import pandas as pd

# criação da base de dados
base = pd.read_csv('census.csv')

# previsores e classe
previsores = base.iloc[:, :14].values
classe = base.iloc[:, 14].values

# transformação de variáveis categóricas (não numéricas)
# colunas 1, 3, 5, 6, 7, 8, 9, 13 e 14 (classe)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
columns = []
for i in range(len(previsores[0])):
    if type(previsores[:, i][0]) == type('str'):
        columns.append(i)
column_tranformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), columns)], remainder='passthrough')
previsores = column_tranformer.fit_transform(previsores).toarray()

labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

'''
from sklearn.preprocessing import LabelEncoder
labelEncoder_previsores = LabelEncoder()
for i in range(len(previsores[0])):
    if type(previsores[:, i][0]) == type('str'):
        previsores[:,i] = labelEncoder_previsores.fit_transform(previsores[:,i])
'''

# padronizando os valores dos previsores
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# separação entre base de treinamento e testes
from sklearn.model_selection import train_test_split
previsoresTreinamento, previsoresTeste, classeTreinamento, classeTeste = train_test_split(previsores,classe,random_state=0)

# treinamento
from keras.models import Sequential
from keras.layers import Dense
classificador = Sequential()
classificador.add(Dense(units = 8, activation = 'relu', input_dim = 108))
classificador.add(Dense(units = 8, activation = 'relu'))
classificador.add(Dense(units = 1, activation = 'sigmoid'))
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classificador.fit(previsoresTreinamento, classeTreinamento, batch_size=10, epochs=100)
resultado = classificador.predict(previsoresTeste)
resultado = (resultado > 0.5)

# análise dos resultados
from sklearn.metrics import confusion_matrix, accuracy_score
matriz = confusion_matrix(classeTeste, resultado)
precisao = accuracy_score(classeTeste, resultado)

