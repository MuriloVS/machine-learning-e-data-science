# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 19:55:20 2020

Algoritmo de Árvore de Decisão: é um algortimo recursivo que
   classifica os atributos em ordem de importância - os mais
   importantes ficam no topo da árvore de decisão. Começa
   calculando a entropia (o quão organizado/desorganizado os
   dados estão) e o ganho de informação para encontrar estes
   atributos. Hoje em dia não é tão usado (random forest
   melhora o desempenho).
   
Vantagens:
 - fácil interpretação (percorrer os nós da árvore);
 - não precisa normalização/padronização dos dados;
 - rápido para classificar novos registros (só percorrer a árvore). 
 
Desvantagens:
 - geração de árvores muito complexas;
 - pequenas mudançãs nos dados podem mudar a árore (poda pode ajudar);
 - problema np-completo para construir a árvore (problema bastante complexo).  
 
 Precisão neste exemplo (75% para treinamento - sem scaler): 0,816975
 Precisão neste exemplo (75% para treinamento - com scaler): 0,816484
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
from sklearn.tree import DecisionTreeClassifier
classificador = DecisionTreeClassifier(criterion='entropy', random_state=0)
classificador.fit(previsoresTreinamento, classeTreinamento)
resultado = classificador.predict(previsoresTeste)

# verificando os resultados
from sklearn.metrics import confusion_matrix, accuracy_score
matriz = confusion_matrix(classeTeste, resultado)
precisao = accuracy_score(classeTeste, resultado)
