#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Questão 1
Modele o problema do Restaurante (mencionado no capítulo 18 do livro texto) 
para ser resolvido por meio de árvore de decisão e por meio de KNN (com k = 1 e com k = 5).

Este código trata só de arvores de devisão
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export
from sklearn.metrics import confusion_matrix, accuracy_score

dados = pd.read_csv("restaurante.csv")

# Fazendo um preprocessamento do dados

# Coloca na variavel entrada só as colunas referentes ao atrubutos
entradas = dados.iloc[:, 0:10].values

classe = dados.iloc[:, 10].values 

# transforma as variáves categoricas em numéricas
labelencoder_entradas = LabelEncoder()
entradas[:, 0] =  labelencoder_entradas.fit_transform(entradas[:, 0])
entradas[:, 1] =  labelencoder_entradas.fit_transform(entradas[:, 1])
entradas[:, 2] =  labelencoder_entradas.fit_transform(entradas[:, 2])
entradas[:, 3] =  labelencoder_entradas.fit_transform(entradas[:, 3])
entradas[:, 4] =  labelencoder_entradas.fit_transform(entradas[:, 4])
entradas[:, 5] =  labelencoder_entradas.fit_transform(entradas[:, 5])
entradas[:, 6] =  labelencoder_entradas.fit_transform(entradas[:, 6])
entradas[:, 7] =  labelencoder_entradas.fit_transform(entradas[:, 7])
entradas[:, 8] =  labelencoder_entradas.fit_transform(entradas[:, 8])
entradas[:, 9] =  labelencoder_entradas.fit_transform(entradas[:, 9])
classe = labelencoder_entradas.fit_transform(classe)




# Normalizado os dados usando a média e o desvio padrão
scaler = StandardScaler()
entradas = scaler.fit_transform(entradas)

#Didvisão dos dados em tetes e treinamento.
entradas_treinamento, entradas_teste, classe_treinamento, classe_teste = train_test_split(entradas, classe, test_size=0.30, random_state=0)

# Instnacioando o classificar da arvores de devisão
# Vai gerar uma arrvores de decisão de acordo com o criterio a entropia

classificador = DecisionTreeClassifier(criterion='entropy')
classificador.fit(entradas_treinamento, classe_treinamento)

#print(classificador.feature_importances_)

# Visualização da arvore
export.export_graphviz(
     classificador,
     out_file = 'arvore.dot',
     feature_names = ['Alternativa', 'Bar', 'Sex/Sab', 'Faminto', 'Clientes', 'Preço', 'Chovendo', 'Reserva', 'Tipo', 'Espera estimada'],
     class_names = ['Sim', 'Não'],
     filled = True,
     leaves_parallel=True
) 


# Usa a base de teste para ver o resultado
resultado_teste = classificador.predict(entradas_teste)

# Verificando a quantidade de acerto nos testes
precisao = accuracy_score(classe_teste, resultado_teste)


print(precisao)
























