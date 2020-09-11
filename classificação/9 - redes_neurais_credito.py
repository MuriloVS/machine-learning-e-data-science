# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 11:56:47 2020

@author: Murilo

Redes Neurais. Utilizadas quando não temos um algorimo específico para resolver
    o problema. Utilizda para descoberta de novos remédios, entendimento de 
    linguagem natural, carros autônomos, reconhecimento facial, cura de doenças,
    bolsa de valores, encontrar soluções para controle de tráfego etc. (muitos
    dados e problemas complexos). Baseado no funcionamento da rede neural de
    um cérebro. Procura imitar o sistema nervoso humano no processo de apren-
    dizagem. Parecido com a troda de informações em uma rede biológica. Com
    Deep Learning as redes neurais ficaram populares novamente.
    
    Fundamentos biológicos: ativação (ou não) do neurônio conforme o potencial
        elétrico do corpo da célula. O neurônio dispara caso a entrada seja
        maior que um número definido.
        
    Neurônio artificial: https://tinyurl.com/yysfswzm
        - Pesos são sinapses;
        - Peso positivo: sinapse excitadora;
        - Peso negetivo: sinapse inibidora;
        - Pesos amplificam ou reduzem o sinal de entrada;
        - Conhecimento da rede neural são os pesos. É este item que a rede pro-
          cura aprender para adequar à base de dados que está sendo analisada.
        
        
        
        
    Vantagens:
        - 
        
    Desvantagens:
        - 
        
Precisão:
  - 
  
    
"""

import pandas as pd

# leitura da base de dados
base = pd.read_csv('credit_data.csv')

# correção dos valores de média negativos (não corrige valores que faltam)
base.loc[base.age < 0, 'age'] = base.loc[base.age > 0, 'age'].mean()
# outra maneira para corrigir valores que faltam (sem usar SimpleImputer)
base.loc[pd.isna(base['age']), 'age'] = base.loc[base.age > 0, 'age'].mean()

# separação entre os previsores e classe
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

# escalonamento
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# separação entre base de treinamento e teste
from sklearn.model_selection import train_test_split
previsoresTreinamento, previsoresTeste, classeTreinamento, classeTeste = train_test_split(previsores, classe, random_state=0)

# treinamento

classificador.fit(previsoresTreinamento, classeTreinamento)
resultado = classificador.predict(previsoresTeste)

from sklearn.metrics import confusion_matrix, accuracy_score
matriz = confusion_matrix(classeTeste, resultado)
precisao = accuracy_score(classeTeste, resultado)
