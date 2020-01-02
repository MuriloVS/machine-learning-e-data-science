# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 20:07:43 2019

@author: Murilo

Algoritmo Naive Bayes: trabalha com uma tabela de probabilidade
Vantagens:
 - rápido;
 - simplicidade de interpratação (multiplicação de probabilidades);
 - trabalha com altas dimensões (muitos atributos);
 - boa previsão para bases pequenas;
 - boma para classificações de textos.
 
Desvantagens:
 - assume que atributos não possuem correlação entre si.
 
Precisão neste exemplo (75% para treinamento): 0,938
    
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

# correção dos valores que faltam
from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
previsores = imputer.fit_transform(previsores)

# padronização dos valores para melhorar o processamento
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# separação entre base de treinamento e testes
from sklearn.model_selection import train_test_split
previsoresTreinamento, previsoresTeste, classeTreinamento, classeTeste = train_test_split(previsores, classe, random_state=0)

# treinando o algoritmo e fazendo a previsão
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsoresTreinamento, classeTreinamento)
resultado = classificador.predict(previsoresTeste)

# análise dos resultados
from sklearn.metrics import confusion_matrix, accuracy_score
matriz = confusion_matrix(classeTeste, resultado)
precisao = accuracy_score(classeTeste, resultado)