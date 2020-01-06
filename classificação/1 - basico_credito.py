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
# mais útil para algoritmos que usam distância euclidiana
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# separação entre base de treinamento e testes
# por padrão 75% dos elementos são usados p/ treinamento e 25% p/ testes
# para alterar podemos utilizar os parâmetros train_size e test_size
# random_state é utilizado como seed, garante que teremos os mesmos resultados em todos as runs
from sklearn.model_selection import train_test_split
previsoresTreinamento, previsoresTeste, classeTreinamento, classeTeste = train_test_split(previsores, classe, random_state=0)

