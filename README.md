# PreditivaFab: previsão de falhas em máquinas com Machine Learning

Projeto avaliativo do Módulo 1 (Desenvolvimento de IA para Análise Preditiva). Montei um pipeline em Python que lê dados de sensores de uma fábrica e tenta prever quando uma máquina vai falhar.

## 1. O problema

Uma fábrica com sensores nas máquinas quer saber com antecedência quando uma delas vai quebrar, para não parar a linha de produção. O modelo precisa prever uma variável com só dois valores:

- `falha_maquina = 1`: a máquina falhou
- `falha_maquina = 0`: a máquina funcionou normalmente

A base tem 10.000 registros de máquinas e apenas cerca de 3,4% são falhas, ou seja, as classes são muito desbalanceadas. Por isso o projeto trata com cuidado o balanceamento dos dados e evita o vazamento de informação (data leakage) entre treino e teste.