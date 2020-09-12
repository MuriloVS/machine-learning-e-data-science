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
        
    Perceptron de uma camada só é adequado para resolução de problemas que são
    linearmente separáveis (uma reta consegue separar as classes). Para pro-
    blemas não lineares utilizamos um perceptron de várias camadas.    
    Multilayer perceptron: https://tinyurl.com/yyhq46rn
    
    Funções de ativação: step (0 ou 1), sigmoid (entre 0 e 1), hyperbolic
    tangent (entre -1 e 1) etc. https://en.wikipedia.org/wiki/Activation_function
    
    Erro: um dos algoritmos mais simples é:
        Uma camada de saída: erro = respostaCorreta - respostaCalculada
        Calcula-se a média absoluta dos erros. 
        Mais de uma cadama de saída:
            erro = (respostaCorreta1 - respostaCalculada1) + 
                   (respostaCorreta2 - respostaCalculada2) + etc.        
        
        Com descida do gradiente: https://tinyurl.com/yxuvozna
        
        Exitem duas técnicas de descida do gradiente para calcular os erros:
            - Batch: cálculo o erro para todos os registros e ajusta todos os
              pesos, conforme necessário. Ttambém temos o mini batch que escolhe
              um número de registros para rodar e atualizar os pesos.
            - Stochastic: cálcula o erro para cada registro individualmente e
              faz o ajuste do peso no registro, conforme necessário. Este tipo
              de descida do gradiente ajuda a prevenir mínimos locais em su-
              perfícies não convexas ( https://tinyurl.com/yybpfecq ). É con-
              siderado mais rápido porque não precisa carregar todos os regis-
              tros na memória (cada um é cálculado individualmente).        
              
              
    Delta de saída: deltaSaida = erro * derivadaSigmoide
        
    Delta da camada escondida: 
        delstaCamadaEscondida = derivadaSigmoide * peso * deltaSaida
    
        
    Backpropagation: atualiza os pesos entre as arestas, do fim para            
        o início. 
        peso(n+1) = (peso*momento)+(entrada*delta*taxaDeAprendizagem)
        Taxa de aprendizagem: quão rápido o algoritmo vai "aprender". Valor alto
            faz com que a convergência ocorra rapidámente mas pode ser que perca-
            se o mínimo global. Com taxa de aprendizagem menor o algoritmo fica
            mais lento, mas com maiores chancer de encontrar o mínimo global cor-
            retamente.
        Momento (momentum): escapar do mínimos locais (nem sempre funciona).
            Quando é alto aumenta a velocidade de convergência; quando é lento
            ajuda a evitar os mínimos locais.
            
    Bias (viés): adiciona um atributo em cada camada, com seus respectivos 
        pesos (em negrito na imagem https://tinyurl.com/y3svn5dy ). Utilizado 
        em muitas arquiteturas de redes neurais. O objetivo é mudar a saída
        com a unidade de bias.    
            
    Em problemas mais complexos é necessário mais que um neurônio na camada de
        saída. ( https://tinyurl.com/yxszaagm )
        Normalmente temos um neurônio para cada classe de saída.
       
    Para deep learning: redes neurais concolucionais, redes neurais recor-
        rentes, Keras, Theano, TensorFlow, programação em GPU etc.
    
    Camadas ocultas: uma das fórmulas mais utlizadas é esta:
        neurônios = (entradas + saídas) / 2
        Onde entradas são as features e as saídas são as classes.
        Lembrando que este é o número de neurônios, não o número de camadas
        ocultas. Duas camadas funcioname bem para poucos dados. Para problemas
        mais complexos, como detecção de câncer através de reconhecimento de
        imagens, mais de 100 camadas podem ser necessárias. Problemas
        linearmente separáveis não necessitam de camadas ocultas.
    
    É possível (recomendável?) utilizar diferentes fórmulas de ativação nas
        camadas ocultas e na camada de saída. Sugestão: duas camadas ocultas
        com função de ativação relu e sigmoide (uma camada de saída) ou
        softmax (mais de uma camada de saída)
        
        
Precisão:
  - Todos os tratamentos (sem alterar parâmetros): 0,994
  - Todos os tratamento (max_iter=1000, tol=0.000001): 0,996
  
  - Todos os tratamento (max_iter=1000, tol=0.000001, activation="tanh"): 0,994
  
  - Todos os tratamento (max_iter=1000, tol=0.000001, activation="logistic"): 0,992
  
  - Todos os tratamento (max_iter=1000, tol=0.000001, solver="lbfgs"): 0,996
  
  - Todos os tratamento (max_iter=1000, tol=0.000001, solver="sgd"): 0,972
  
  - Sem escalonamento (sem alterar parâmetros): 0,872
  
    
"""

import pandas as pd, numpy as np

# leitura da base de dados
base = pd.read_csv('credit_data.csv')

# correção dos valores de média negativos (não corrige valores que faltam)
base.loc[base.age < 0, 'age'] = base.loc[base.age > 0, 'age'].mean()

# separação entre os previsores e classe
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
previsores = imputer.fit_transform(previsores)

# escalonamento
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# separação entre base de treinamento e teste
from sklearn.model_selection import train_test_split
previsoresTreinamento, previsoresTeste, classeTreinamento, classeTeste = train_test_split(previsores, classe, random_state=0)

# treinamento
# MLPC = Multi-layer perceptron
from sklearn.neural_network import MLPClassifier
classificador = MLPClassifier(verbose=True,
                              max_iter=1000,
                              tol=0.000001,
                              solver="adam",
                              activation="relu",
                              hidden_layer_sizes=(100))
classificador.fit(previsoresTreinamento, classeTreinamento)
resultado = classificador.predict(previsoresTeste)

from sklearn.metrics import confusion_matrix, accuracy_score
matriz = confusion_matrix(classeTeste, resultado)
precisao = accuracy_score(classeTeste, resultado)
