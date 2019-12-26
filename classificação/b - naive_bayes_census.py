# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 22:32:42 2019

@author: Murilo

Com pré-processametos:
Precisão neste exemplo (60% para treinamento): 0,5587715
Precisão neste exemplo (75% para treinamento): 0,5600049
Precisão neste exemplo (80% para treinamento): 0,5584216
Precisão neste exemplo (90% para treinamento): 0,5594105

Sem Scaler e LabelEncoder (60% para treinamento) :0,7963147
Sem Scaler e LabelEncoder (75% para treinamento) :0,7949883
Sem Scaler e LabelEncoder (90% para treinamento) :0,7884556
"""

import pandas as pd

# criação da base de dados
base = pd.read_csv('census.csv')

# previsores e classe
previsores = base.iloc[:, :14].values
classe = base.iloc[:, 14].values

# outra maneira para fazer as adequações dos valores
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
oneHotEncorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1,3,5,6,7,8,9,13])], remainder='passthrough')
previsores = oneHotEncorder.fit_transform(previsores).toarray()

# padronizando os valores dos previsores
# foi feita a opção de não padronizar as colunas criadas pelo OneHotEncoder
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores[:, 102:] = scaler.fit_transform(previsores[:, 102:])

# transformando os valores da classe em numéricos
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
classe = encoder.fit_transform(classe)

# separação entre base de treinamento e testes
from sklearn.model_selection import train_test_split
previsoresTreinamento, previsoresTeste, classeTreinamento, classeTeste = train_test_split(previsores, classe, train_size=0.75, random_state=0)

# treinando o algoritmo e fazendo a previsão
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsoresTreinamento, classeTreinamento)
resultado = classificador.predict(previsoresTeste)

# análise dos resultados
from sklearn.metrics import confusion_matrix, accuracy_score
matriz = confusion_matrix(classeTeste, resultado)
precisao = accuracy_score(classeTeste, resultado)
