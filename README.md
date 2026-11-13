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


## 3. Estrutura do repositório

```
manutencao-preditiva-ml/
├── data/
│   └── manutencao_preditiva.csv      # base de dados
├── docs/
│   └── anotações do Departamento de Engenharia.docx
├── projeto_manutencao_preditiva.ipynb  # notebook principal (7 fases)
├── requirements.txt                    # dependências com versões
└── README.md
```

## 4. Como executar

Pré-requisito: Python 3.14 e Git instalados.

1. Clone o repositório e entre na pasta:

```powershell
git clone https://github.com/arthursphilippi-coder/manutencao-preditiva-ml.git
cd manutencao-preditiva-ml
```

2. Crie e ative um ambiente virtual (Windows PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

3. Instale as dependências:

```powershell
pip install -r requirements.txt
```

4. Abra o notebook `projeto_manutencao_preditiva.ipynb` no VS Code ou no Jupyter, selecione o kernel do ambiente `.venv` e execute todas as células (**Run All**).

Os dados são lidos por caminho relativo (`data/manutencao_preditiva.csv`), então o notebook funciona em qualquer máquina sem ajustes.

## 5. Resultados

Acurácia no conjunto de teste (2.000 registros), com os melhores parâmetros de cada algoritmo:

| Modelo | Configuração | Acurácia no teste |
|---|---|---|
| KNN | K = 3 | 90,45% |
| Árvore de Decisão | `max_depth = None` | 92,80% |

**Veredito:** a Árvore de Decisão teve a maior acurácia no teste e é o modelo recomendado.

**Overfitting:** a árvore sem limite de profundidade chegou a 33 níveis e acertou 99,26% do treino contra 92,80% do teste. Com `max_depth = 5`, treino e teste ficaram praticamente iguais (cerca de 89,95%), sendo a configuração mais estável.

**Ressalva sobre a métrica:** sempre prever "sem falha" já acertaria 96,6% do teste, então a acurácia sozinha não mede bem a detecção de falhas. A matriz de confusão mostra que os modelos detectam cerca de 65% das falhas (recall de 0,647 no KNN e 0,662 na árvore).

## 6. Melhorias possíveis

- Usar validação cruzada para escolher os hiperparâmetros, deixando o teste só para a avaliação final.
- Usar `SMOTENC`, versão do SMOTE para variáveis categóricas, já que `tipo_cod` é categórica.
- Calcular as médias e medianas da imputação só com os dados de treino, evitando um vazamento pequeno de informação.
- Adotar recall e F1-score como métricas principais, pois o custo de uma falha não detectada é maior que o de um alarme falso.
- Testar outros algoritmos, como Random Forest e Gradient Boosting.

## 7. Organização do Git

- `main`: versão final do projeto.
- `develop`: integração das funcionalidades.
- Uma branch por tarefa, criada a partir da `develop`: `feature/setup`, `feature/eda`, `feature/data-prep`, `feature/feature-engineering`, `feature/split-balanceamento`, `feature/escalonamento`, `feature/tuning`, `feature/avaliacao-final` e `feature/readme`.
- Cada branch foi integrada por Pull Request e mantida no repositório.