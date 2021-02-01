# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:54:16 2021

@author: mvs_r
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 21:54:01 2021

Utilizando k cross validation estratificada. Este método procura fazer uma melhor
divisão entre as classes nos folds criados (estamos usando 10). As partes repre-
sentam melhor o todo.

A precisão ficou em 0,9254 e o desvio padrão em 0,0136

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

# criamos um vetor coluna para auxiliar a utilização dos indíces no treinamento
import numpy as np
aux = np.zeros(shape=(previsores.shape[0], 1))  

# importando a biblioteca para fazer a validação cruzada estratificada, com k=10
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
resultado = []
matrizes = []
for indiceTreinamento, indiceTeste in kfold.split(previsores, aux):
    classificador.fit(previsores[indiceTreinamento], classe[indiceTreinamento])
    previsoes = classificador.predict(previsores[indiceTeste])
    precisao = accuracy_score(classe[indiceTeste], previsoes)
    resultado.append(precisao)
    matrizes.append(confusion_matrix(classe[indiceTeste], previsoes))

resultado = np.asarray(resultado)
resultado.mean()
resultado.std()
matrizFinal = np.mean(matrizes, axis=0)