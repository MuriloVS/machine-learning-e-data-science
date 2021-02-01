# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 17:51:43 2021

Salvando os 3 melhores modelos (random forest, SVM e redes neurais).

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

# Random Forest
from sklearn.ensemble import RandomForestClassifier
classificadorRF = RandomForestClassifier(n_estimators=40, criterion='entropy')
classificadorRF.fit(previsores, classe)

# SVM
from sklearn.svm import SVC
classificadorSVM = SVC(kernel='rbf', C=2.0)
classificadorSVM.fit(previsores, classe)

# redes neurais
from sklearn.neural_network import MLPClassifier
classificadorRN = MLPClassifier(verbose=False, max_iter=1000, tol=0.0001,
                                solver='adam', hidden_layer_sizes=(100),
                                activation='relu', batch_size=200, 
                                learning_rate_init=0.001)
classificadorRN.fit(previsores, classe)

import pickle
pickle.dump(classificadorRF, open('RF_final.sav', 'wb'))
pickle.dump(classificadorSVM, open('SVM_final.sav', 'wb'))
pickle.dump(classificadorRN, open('RN_final.sav', 'wb'))