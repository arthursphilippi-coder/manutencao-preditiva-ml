# PreditivaFab: previsão de falhas em máquinas com Machine Learning

Projeto avaliativo do Módulo 1 (Desenvolvimento de IA para Análise Preditiva). Montei um pipeline em Python que lê dados de sensores de uma fábrica e tenta prever quando uma máquina vai falhar.

## 1. O problema

Uma fábrica com sensores nas máquinas quer saber com antecedência quando uma delas vai quebrar, para não parar a linha de produção. O modelo precisa prever uma variável com só dois valores:

- `falha_maquina = 1`: a máquina falhou
- `falha_maquina = 0`: a máquina funcionou normalmente

A base tem 10.000 registros de máquinas e apenas cerca de 3,4% são falhas, ou seja, as classes são muito desbalanceadas. Por isso o projeto trata com cuidado o balanceamento dos dados e evita o vazamento de informação (data leakage) entre treino e teste.


## 2. Técnicas e tecnologias

**Linguagem e bibliotecas:** Python 3.14, pandas, NumPy, Matplotlib, Seaborn, scikit-learn e imbalanced-learn (versões exatas no requirements.txt).

**Etapas do pipeline (todas no notebook `projeto_manutencao_preditiva.ipynb`):**

| Fase | O que foi feito |
|---|---|
| 1. Análise exploratória | Dimensões, tipos, `describe()`, histogramas, gráfico do desbalanceamento da variável alvo e mapa de calor de correlação de Pearson |
| 2. Limpeza de dados | Remoção de duplicados, tratamento de valores ausentes (média nas temperaturas e no torque, mediana na rotação) e boxplots com regra do IQR para outliers |
| 3. Feature engineering | Nova coluna `potencia = velocidade_rotacao_rpm * torque_nm` |
| 4. Divisão e balanceamento | Separação X e y, divisão 80/20 com `stratify=y` e SMOTE aplicado somente no treino |
| 5. Escalonamento | `StandardScaler` apenas para o KNN (`fit_transform` no treino e `transform` no teste); árvore sem escalonar |
| 6. Ajuste de parâmetros | KNN com diferentes valores de K e Árvore de Decisão com diferentes valores de `max_depth`, comparando acurácia de treino e teste para identificar overfitting |
| 7. Avaliação final | Acurácia dos melhores modelos no teste, matriz de confusão, recall e veredito |

**Decisões importantes:**

- As colunas `falha_twf`, `falha_hdf`, `falha_pwf`, `falha_osf` e `falha_rnf` ficaram fora das variáveis preditoras, pois só são conhecidas depois que a falha acontece (vazamento de dados).
- A coluna `tipo` (L, M, H) foi codificada de forma ordinal (0, 1, 2).
- O SMOTE foi aplicado depois da divisão e só no treino; o teste mantém a distribuição real.