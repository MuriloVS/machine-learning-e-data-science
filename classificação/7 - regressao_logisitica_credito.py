# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 18:47:43 2020

@author: Murilo

Algoritmo de Regressão Logística: apesar do nome, este é um algoritmo de
   classificação (trabalha com rótulos) e não um algoritmo de regressão
   (trabalha com números). Exemplo: uma pessoa vaiu pagar um empréstimo?
   Algoritmo de classificação (a resposta, ou o que quer ser previsto, se
   resume aos rótulo 'sim' e 'não'). Qual o limite de crédito deste cliente?
   500? 1000? Estes rótulos são numéricos, seria necessário usar um método/
   algoritmo de predição de regressão.
   Encontra e melhor função sigmóide para os dados.
   Retorna valores entre 0 e 1 (probabilidade).

Precisão:
  - Todos os tratamentos: 0,946
  - Sem StandardScaler: 0,93  
   
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
from sklearn.linear_model import LogisticRegression
classificador = LogisticRegression(random_state=0)
classificador.fit(previsoresTreinamento, classeTreinamento)
resultado = classificador.predict(previsoresTeste)

# análise dos resultados
from sklearn.metrics import confusion_matrix, accuracy_score
matriz = confusion_matrix(classeTeste, resultado)
precisao = accuracy_score(classeTeste, resultado)

