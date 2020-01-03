# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 12:28:29 2020

@author: Murilo

Algoritmos de aprendizagem por regras (similar aos algoritmos
   de árvore de decisão).
"""

import Orange

base = Orange.data.Table('census.csv')
baseDividida = Orange.evaluation.testing.sample(base, n=0.25)
baseTreinamento = baseDividida[1]
baseTeste= baseDividida[0]

cn2_learner = Orange.classification.rules.CN2Learner()
classificador = cn2_learner(baseTreinamento)

resultado = Orange.evaluation.testing.TestOnTestData(baseTreinamento, baseTeste, [classificador])
Orange.evaluation.CA(resultado)