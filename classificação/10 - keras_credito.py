# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 22:11:28 2020

@author: mvs_r

Keras.  
        
Precisão (os dados variam - melhor resultado anotado):
  - Todos os tratamentos (batch_size=10, epochs=100): 0,998
  - Todos os tratamentos (batch_size=10, epochs=100): 1,000
  
  - Sem escalonamento (batch_size=10, epochs=100): 0,872
  
    
"""

import pandas as pd

# leitura da base de dados
base = pd.read_csv('credit_data.csv')

# correção dos valores de média negativos (não corrige valores que faltam)
base.loc[base.age < 0, 'age'] = base.loc[base.age > 0, 'age'].mean()

# separação entre os previsores e classe
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values
# corrige os valores que faltam
from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
previsores = imputer.fit_transform(previsores)

# escalonamento
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# separação entre base de treinamento e teste
from sklearn.model_selection import train_test_split
previsoresTreinamento, previsoresTeste, classeTreinamento, classeTeste = train_test_split(previsores, classe, random_state=0)

# treinamento
from keras.models import Sequential
from keras.layers import Dense
classificador = Sequential()
# units -> neurônios = (entrda + saída) / 2
# input_dim -> entradas = 3, neste caso
# criação das camadas de entrada, oculta e de saída
classificador.add(Dense(units=2, activation="relu", input_dim=3))
classificador.add(Dense(units=2, activation="relu"))
classificador.add(Dense(units=1, activation="sigmoid"))  # sigmoid pq a saída é binária
classificador.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
# batch_size -> descida do gradiente estocástica (será realizo o ajuste de 10 em 10)
# epochs -> número de vezes que será executada
classificador.fit(previsoresTreinamento, classeTreinamento, batch_size=10, epochs=100)
resultado = classificador.predict(previsoresTeste)
resultado = (resultado > 0.5)

from sklearn.metrics import confusion_matrix, accuracy_score
matriz = confusion_matrix(classeTeste, resultado)
precisao = accuracy_score(classeTeste, resultado)
