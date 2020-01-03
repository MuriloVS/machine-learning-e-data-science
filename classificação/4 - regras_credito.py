# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 22:02:13 2020

@author: Murilo

Algoritmos de aprendizagem por regras (similar aos algoritmos
   de árvore de decisão).

Precisão neste exemplo: 0,5
"""

import Orange

base = Orange.data.Table('credit_data.csv')

baseDividida = Orange.evaluation.testing.sample(base, n=0.25)
baseTreinamento = baseDividida[1]
baseTeste = baseDividida[0]

cn2_learner = Orange.classification.rules.CN2Learner()
classificador = cn2_learner(baseTreinamento)

resultado = Orange.evaluation.testing.TestOnTestData(baseTreinamento, baseTeste, [classificador])
Orange.evaluation.CA(resultado)

#for regras in classificador.rule_list:
#    print(regras)
