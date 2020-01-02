# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 18:28:31 2020

@author: Murilo

Algoritmo de Árvore de Decisão: é um algortimo recursivo que
   classifica os atributos em ordem de importância - os mais
   importantes ficam no topo da árvore de decisão. Começa
   calculando a entropia (o quão organizado/desorganizado os
   dados estão) e o ganho de informação para encontrar estes
   atributos. Hoje em dia não é tão usado (random forest
   melhora o desempenho).
   
Vantagens:
 - fácil interpretação (percorrer os nós da árvore);
 - não precisa normalização/padronização dos dados;
 - rápido para classificar novos registros (só percorrer a árvore). 
 
Desvantagens:
 - geração de árvores muito complexas;
 - pequenas mudançãs nos dados podem mudar a árore (poda pode ajudar);
 - problema np-completo para construir a árvore (problema bastante complexo).  
 
 Precisão neste exemplo (75% para treinamento): 0,982
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

# separação entre base de treinamento e teste
from sklearn.model_selection import train_test_split
previsoresTreinamento, previsoresTeste, classeTreinamento, classeTeste = train_test_split(previsores, classe, random_state=0)

# treinamento
from sklearn.tree import DecisionTreeClassifier
classificador = DecisionTreeClassifier(criterion='entropy', random_state=0)
classificador.fit(previsoresTreinamento, classeTreinamento)
resultado = classificador.predict(previsoresTeste)

# análise dos resultados
from sklearn.metrics import confusion_matrix, accuracy_score
matriz = confusion_matrix(classeTeste, resultado)
precisao = accuracy_score(classeTeste, resultado)
