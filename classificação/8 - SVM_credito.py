# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 11:56:47 2020

@author: Murilo

Algoritmo SVM (Support Vector Machines). Em geral supera outros algoritmos
    de aprendizagem de máquina. Utilizado para tarefas complexas (reconhecimento
    de voz, imagens, caracteres etc.). Aprende hiperplanos de separação com
    margem máxima (reta separa as classes em um plano - parecido com regressão
    logística, mas com margem máxima das pontos mais próximos até a reta).    
    Exemplo: https://tinyurl.com/y6l2phjm
    
    Um dos métodos para encontrar hiperplano com a reta de máxima distância é
    chamada de Convex Hulls (envoltória/casca convexa).    
    Convex hulls: https://tinyurl.com/yxrpjdn5
    
    Outro método, um dos mais utilizados no momento, é a abordagem matemática.    
    Abordagem matemática: https://tinyurl.com/yxzjl8yd
    
    Para problemas não lineares é utilizado um método chamado Kernel Trick.    
    Kernel Trick: https://tinyurl.com/yxrpdv5b

    Vantagens:
        - não é muito influenciado por ruídos nos dados;
        - utilizado para classificação e regressão;
        - aprende conceitos não presentes nos dados originais;
        - mais fácil de usar que redes neurais;
        
    Desvantagens:
        - necessário testar várias combinações de parâmetros;
        - lento (cálculos numéricos complexos);
        - black box (não é possível visualizar o que é gerado (regras, tabelas etc.))
        
Precisão:
  - Todos os tratamentos (linear): 0,946
  - Sem StandardScaler (linear): 0,944 - cálculos consideravelmente mais lentos
  
  - Todos os tratamentos (rbf): 0,982
  - Todos os tratamentos (rbf, C=0.9): 0,98
  - Todos os tratamentos (rbf, C=2): 0,988
  - Todos os tratamentos (rbf, C=3): 0,984
  - Todos os tratamentos (rbf, C=4): 0,986
  - Todos os tratamentos (rbf, C=10): 0,988
  - Sem StandardScaler (rbf): 0,872
  
  - Todos os tratamentos (poly): 0,968
  - Sem StandardScaler (poly): 0,872
  
  - Todos os tratamentos (sigmoid): 0,838
  - Sem StandardScaler (sigmoid): 0,862
  
    
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
from sklearn.svm import SVC
classificador = SVC(kernel='rbf', random_state=1, C=10)
classificador.fit(previsoresTreinamento, classeTreinamento)
resultado = classificador.predict(previsoresTeste)

from sklearn.metrics import confusion_matrix, accuracy_score
matriz = confusion_matrix(classeTeste, resultado)
precisao = accuracy_score(classeTeste, resultado)
