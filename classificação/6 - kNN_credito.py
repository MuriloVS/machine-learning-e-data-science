# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 16:23:09 2020

@author: Murilo

Algoritmo kNN (k-Nearest Neighbour). Calcula a distância de um ponto
   até os demais pontos já existentes. Se o valor de k for igual a um,
   classificamos o novo registro baseado no ponto com a menor distância.
   Se k for igual a dois, utilizamos os dois pontos com menores distâncias
   e assim por diante. Em caso de empate existem critérios para resolver a 
   classificação do registro. NÃO geram modelo (lazy). As variáveis
   PRECISAM estar na mesma escala.

Características:   
  - indicado quando o relacionamento entre as caracterpisticas é complexo;
  - valor de k pequeno: dados com ruídos e outliers pode prejudicar;
  - valor de k grande: tendência a classificar a classe com mais elementos
    (overfitting) - valor default 3 ou 5;
  - lento para fazer previsões - recalcular a cada novo registro;
  - além de distância euclidiana: coeficiente de Pearson, índice de Tanimoto,
    City Block.
     
Precisão:
  - Todos os tratamentos: 0,986
  - Sem StandardScaler: 0,836
"""

import pandas as pd

# leitura da base de dados
base = pd.read_csv('credit_data.csv')

# correção de valores incorretos
ageMedia = base.loc[base.age > 0, 'age'].mean()
base.loc[base.age < 0, 'age'] = ageMedia

# separação entre previsores e classe
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

# correção dos valores faltantes
from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
previsores[:, 1:4] = imputer.fit_transform(previsores[:, 1:4])

# escalonamento
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# separação entre base de treinamento e teste
from sklearn.model_selection import train_test_split
previsoresTreinamento, previsoresTeste, classeTreinamento, classeTeste = train_test_split(previsores, classe, random_state=0)

# treinamento
from sklearn.neighbors import KNeighborsClassifier
classificador = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classificador.fit(previsoresTreinamento, classeTreinamento)
resultado = classificador.predict(previsoresTeste)

# análise dos resultados
from sklearn.metrics import confusion_matrix, accuracy_score
matriz = confusion_matrix(classeTeste, resultado)
precisao = accuracy_score(classeTeste, resultado)
