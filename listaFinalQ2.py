#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd

dados = pd.read_csv("pima.csv")

# Fazendo um preprocessamento do dados

# Coloca na variavel entrada só as colunas referentes ao atrubutos
entradas = dados.iloc[:, 0:8].values

classe = dados.iloc[:, 8].values 

# Normalizado os dados usando a média e o desvio padrão
scaler = StandardScaler()
entradas = scaler.fit_transform(entradas)

# Divisão dos dados em treinamento e testes
entradas_treinamento, entradas_teste, classe_treinamento, classe_teste = train_test_split(entradas, classe, test_size=0.30, random_state=0)
    
# Criação da rede neural
rede = MLPClassifier(
           activation='logistic',
           learning_rate_init=0.3,
           max_iter=200,
           random_state=1,
           verbose=200
           
        )
rede.fit(entradas_treinamento, classe_treinamento)

# Usando os dados de testes para veirficar os resultados

resultados = rede.predict(entradas_teste) 

# Comparando os resultados com o esperado

precisao = accuracy_score(classe_teste, resultados)
print(precisao)    
    
    
    