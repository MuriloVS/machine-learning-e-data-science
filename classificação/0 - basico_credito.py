# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import pandas as pd
import numpy as np

# criando a base de dados
base = pd.read_csv('credit_data.csv')

# trantando os erros encontrados na coluna 'age'
ageMedia = base.loc[base.age > 0, 'age'].mean()
base.loc[base.age < 0, 'age'] = ageMedia
base.loc[pd.isnull(base.age) == True, 'age'] = ageMedia

# criando os previsores e a classe (resultado a ser analisado)
# iloc[linhas, colunas]
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

# outra maneira de lidar com valores faltantes em 'age'
from sklearn.impute import SimpleImputer
# os parâmetros passados em SimpleImputer podem ser ignorados
# (são os valores padrão)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
previsores = imputer.fit_transform(previsores)

# escalonamento dos atributos
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
