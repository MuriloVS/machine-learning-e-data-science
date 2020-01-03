# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 21:22:00 2020

@author: Murilo

Algoritmo de Árvore de Decisão com Random Forest: neste caso
  temos a combinação de diversas árvores de decisão (ensemble
  learning - aprendizagem em conjunto). Escolhe de forma alea
  tória 'K' atributos para a comparação da métrica de pureza/
  impureza (impureza de gini/entropia). "Vários algortimos juntos
  para construir um melhor." Utiliza-se média quando estamos tra-
  balhando com regressão e "votos da maioria" em classificação
  para obter a resposta final.

Precisão neste exemplo (n_estimators=10, sem onehotencoder e scaler): 0,84535
Precisão neste exemplo (n_estimators=50, sem onehotencoder e scaler): 0,85222
Precisão neste exemplo (n_estimators=10): 0,84301
Precisão neste exemplo (n_estimators=50): 0,84989
"""

import pandas as pd

# criação da base de dados
base = pd.read_csv('census.csv')

# separação entre previsores e classe
previsores = base.iloc[:, :14].values
classe = base.iloc[:, 14].values

# transformando atributos categóricos para numéricos
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
for i in range(len(previsores[0])):
    if type(previsores[:, i][0]) == type('str'):
        previsores[:, i] = labelEncoder.fit_transform(previsores[:, i])
        
# outra maneira para fazer as adequações dos valores
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
oneHotEncorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1,3,5,6,7,8,9,13])], remainder='passthrough')
previsores = oneHotEncorder.fit_transform(previsores).toarray()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores[:, 102:] = scaler.fit_transform(previsores[:, 102:])
        
# separação entre base de dados de treinamento e teste
from sklearn.model_selection import train_test_split
previsoresTreinamento, previsoresTeste, classeTreinamento, classeTeste = train_test_split(previsores, classe, random_state=0)

# treinamento
from sklearn.ensemble import RandomForestClassifier
classificador = RandomForestClassifier(n_estimators=10,criterion='entropy', random_state=0)
classificador.fit(previsoresTreinamento, classeTreinamento)
resultado = classificador.predict(previsoresTeste)

# verificando os resultados
from sklearn.metrics import confusion_matrix, accuracy_score
matriz = confusion_matrix(classeTeste, resultado)
precisao = accuracy_score(classeTeste, resultado)
