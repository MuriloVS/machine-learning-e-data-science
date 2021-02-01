# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:54:16 2021

@author: mvs_r
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 21:54:01 2021

Testando cada método 30 vezes (10 folds, cada). Resultados em planilha do Excel.

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

# from sklearn.naive_bayes import GaussianNB
# classificador = GaussianNB()

# from sklearn.tree import DecisionTreeClassifier
# classificador = DecisionTreeClassifier()

# from sklearn.ensemble import RandomForestClassifier
# classificador = RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=0)

# from sklearn.neighbors import KNeighborsClassifier
# classificador = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

# from sklearn.linear_model import LogisticRegression
# classificador = LogisticRegression(random_state=0)

# from sklearn.svm import SVC
# classificador = SVC(kernel='rbf', random_state=1, C=10)

from sklearn.neural_network import MLPClassifier
classificador = MLPClassifier(verbose=False,
                              max_iter=1000,  
                              tol=0.00001,
                              solver="adam",
                              activation="relu",
                              hidden_layer_sizes=(100))

# criamos um vetor coluna para auxiliar a utilização dos indíces no treinamento
import numpy as np
aux = np.zeros(shape=(previsores.shape[0], 1))  

# importando a biblioteca para fazer a validação cruzada estratificada, com k=10
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
resultadoFinal = []

for i in range(1, 31):    
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
    resultado = []
    
    for indiceTreinamento, indiceTeste in kfold.split(previsores, aux):
        classificador.fit(previsores[indiceTreinamento], classe[indiceTreinamento])
        previsoes = classificador.predict(previsores[indiceTeste])
        precisao = accuracy_score(classe[indiceTeste], previsoes)
        resultado.append(precisao)
    
    resultado = np.asarray(resultado)
    resultadoFinal.append(resultado.mean())

for i in range(len(resultadoFinal)):
    print(str(resultadoFinal[i]).replace('.', ','))