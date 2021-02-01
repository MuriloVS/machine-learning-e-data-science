# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 18:10:14 2021

@author: mvs_r
"""

import pandas as pd
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np

base = pd.read_csv('credit_data.csv')
base.loc[base.age < 0, 'age'] = 40.92              
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

SVM = pickle.load(open('SVM_final.sav', 'rb'))
RF = pickle.load(open('RF_final.sav', 'rb'))
RN = pickle.load(open('RN_final.sav', 'rb'))

resultado_SVM = SVM.score(previsores, classe)
resultado_RF = RF.score(previsores, classe)
resultado_RN = RN.score(previsores, classe)

novo_registro = np.asarray([[20000, 20, 15000]])
novo_registro = novo_registro.reshape(-1, 1)
novo_registro = scaler.fit_transform(novo_registro)
novo_registro = novo_registro.reshape(-1, 3)

resposta_SVM = SVM.predict(novo_registro)
resposta_RF = RF.predict(novo_registro)
resposta_RN = RN.predict(novo_registro)