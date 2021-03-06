# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 22:57:29 2019

@author: Murilo
"""

import pandas as pd

# criação da base de dados
base = pd.read_csv('census.csv')
# base.describe()

# previsores e classe
previsores = base.iloc[:, :14].values
classe = base.iloc[:, 14].values

# transformação de variáveis categóricas (não numéricas)
# colunas 1, 3, 5, 6, 7, 8, 9, 13 e 14
from sklearn.preprocessing import LabelEncoder
labelEncoder_previsores = LabelEncoder()
previsores[:,1] = labelEncoder_previsores.fit_transform(previsores[:,1])
previsores[:,3] = labelEncoder_previsores.fit_transform(previsores[:,3])
previsores[:,5] = labelEncoder_previsores.fit_transform(previsores[:,5])
previsores[:,6] = labelEncoder_previsores.fit_transform(previsores[:,6])
previsores[:,7] = labelEncoder_previsores.fit_transform(previsores[:,7])
previsores[:,8] = labelEncoder_previsores.fit_transform(previsores[:,8])
previsores[:,9] = labelEncoder_previsores.fit_transform(previsores[:,9])
previsores[:,13] = labelEncoder_previsores.fit_transform(previsores[:,13])

labelEncoder_classe = LabelEncoder()
classe = labelEncoder_classe.fit_transform(classe)

# evita que seja atribuído um peso às variáveis nominais, após sua transformação
# transforma uma coluna em várias com valores de 0 a 1, conforme necessário
from sklearn.preprocessing import OneHotEncoder
oneHotEncoder = OneHotEncoder(categorical_features=[1, 3, 5, 6, 7, 8, 9, 13])
previsores = oneHotEncoder.fit_transform(previsores).toarray()

# outra maneira para fazer as adequações dos valores
from sklearn.compose import ColumnTransformer
oneHotEncorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1,3,5,6,7,8,9,13])], remainder='passthrough')
previsores = oneHotEncorder.fit_transform(previsores).toarray()

# padronizando os valores dos previsores
# foi feita a opção de não padronizar as colunas criadas pelo OneHotEncoder
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores[:, 102:] = scaler.fit_transform(previsores[:, 102:])

# separação entre base de treinamento e testes
# por padrão 75% dos elementos são usados p/ treinamento e 25% p/ testes
# para alterar podemos utilizar os parâmetros train_size e test_size
# random_state é utilizado como seed, garante que teremos os mesmos resultados em todos as runs
from sklearn.model_selection import train_test_split
previsoresTreinamento, previsoresTeste, classeTreinamento, classeTeste = train_test_split(previsores, classe, random_state=0)
