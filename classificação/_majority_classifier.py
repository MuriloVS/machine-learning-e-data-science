# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 16:05:12 2020

Majority Learner classifica os dados de acordo com a maioria dos resultados
   analisados. Serve como um "base line classifier" - se os resultados de um
   algoritmo ficarem abaixo do resultado encontrado, é mais fácil simplesmente
   usar este classificador. Neste caso o algoritmo encontrou um valor aproxi-
   mado de 0,85 para a base de crédito e 0,76 para o census.

@author: Murilo
"""

import Orange

base = Orange.data.Table('credit_data.csv')

baseDividida = Orange.evaluation.testing.sample(base, n=0.25)
baseTreinamento = baseDividida[1]
baseTeste = baseDividida[0]

classificador = Orange.classification.MajorityLearner()
resultado = Orange.evaluation.testing.TestOnTestData(baseTreinamento, baseTeste, [classificador])

Orange.evaluation.CA(resultado)

base = Orange.data.Table('census.csv')

baseDividida = Orange.evaluation.testing.sample(base, n=0.25)
baseTreinamento = baseDividida[1]
baseTeste = baseDividida[0]

classificador = Orange.classification.MajorityLearner()
resultado = Orange.evaluation.testing.TestOnTestData(baseTreinamento, baseTeste, [classificador])

Orange.evaluation.CA(resultado)

