# -*- coding: utf-8 -*-

"""
Created on Wed Jan  2 20:20:31 2020

@author: Murilo

Algoritmo de Árvore de Decisão com Random Forest: neste caso
  temos a combinação de diversas árvores de decisão (ensemble
  learning - aprendizagem em conjunto). Escolhe de forma alea
  tória 'K' atributos para a comparação da métrica de pureza/
  impureza (impureza de gini/entropia). "Vários algortimos juntos
  para construir um melhor." Utiliza-se média quando estamos tra-
  balhando com regressão e "votos da maioria" em classificação
  para obter a resposta final.
   
Vantagens:
 - 
 
Desvantagens:
 -
 
 Precisão neste exemplo (n_estimators=10, com e sem scaler): 0,968
 Precisão neste exemplo (n_estimators=20, com e sem scaler): 0,978
 Precisão neste exemplo (n_estimators=50, com e sem scaler): 0,98
 
 * Scaler não é necessário.
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

# escalonamento (testar para ver se há melhora)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# separação entre base de treinamento e teste
from sklearn.model_selection import train_test_split
previsoresTreinamento, previsoresTeste, classeTreinamento, classeTeste = train_test_split(previsores, classe, random_state=0)

# treinamento
from sklearn.ensemble import RandomForestClassifier
classificador = RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=0)
classificador.fit(previsoresTreinamento, classeTreinamento)
resultado = classificador.predict(previsoresTeste)

# análise dos resultados
from sklearn.metrics import confusion_matrix, accuracy_score
matriz = confusion_matrix(classeTeste, resultado)
precisao = accuracy_score(classeTeste, resultado)
