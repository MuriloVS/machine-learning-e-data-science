# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 21:54:01 2021

Utilizando k cross validation em vez de train_test_split com o Naive Bayes.
Esse método é o mais usado pela comunidade científica e funciona da seguinte forma:
    - k é em quantas partes a base de dados vai ser dividida;
    - cada parte é usada tanto para teste como para treinamento;
    - k = 10 é o padrão;
    - ex.: k(1) para teste, outrs 9 para treinamento; depois k(2) para teste,
      os outros 9 para treinamento (inlusive a primeira parte. O resultado final
      é uma média dos 10 resuldados.
    - neste exemplo a média das 10 execuções ficou em 0,924 e desvio padrão
      de 0.020 - desvio padrão muito alto pode indicar overfitting (quando o
      modelo se ajusta muito bem aos dados de treinamento mas não tem resul-
      tado tão bom com entradas novas)    

@author: mvs_r
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

from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()

# importando a biblioteca para fazer a validação cruzada, com k=10
from sklearn.model_selection import cross_val_score
resultados = cross_val_score(classificador, previsores, classe, cv = 10)
resultado = resultados.mean()
