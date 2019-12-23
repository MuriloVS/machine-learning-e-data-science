# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import pandas as pd

# criando a base de dados
base = pd.read_csv('credit_data.csv')

# trantando os erros encontrados na coluna 'age'
ageMedia = base.loc[base.age > 0, 'age'].mean()
base.loc[base.age < 0, 'age'] = ageMedia
base.loc[pd.isnull(base.age) == True, 'age'] = ageMedia

# criando os previsores e a classe (resultado a ser analisado)
# iloc[linhas, colunas]
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4]

from sklearn.impute import SimpleImputer

imputer = SimpleImputer()
previsores = imputer.fit_transform(previsores)
