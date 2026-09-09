
<img src="../assets/logo-fiap.png" alt="FIAP - Faculdade de Informática e Admnistração Paulista" border="0" width=30% height=30%>

# AI Project Document - Módulo 1 - FIAP

## FarmTech Solutions

#### Nomes dos integrantes do grupo

Renan Ramos — RM573201

## Sumário

[1. Introdução](#c1)

[2. Visão Geral do Projeto](#c2)

[3. Desenvolvimento do Projeto](#c3)

[4. Resultados e Avaliações](#c4)

[5. Conclusões e Trabalhos Futuros](#c5)

[6. Referências](#c6)

[Anexos](#c7)

<br>

# <a name="c1"></a>1. Introdução

## 1.1. Escopo do Projeto

### 1.1.1. Contexto da Inteligência Artificial

A Inteligência Artificial aplicada ao agronegócio (AgriTech) combina dados de clima, solo e manejo para apoiar decisões de produção. No Brasil, o setor é estratégico: a fazenda de médio porte atendida neste projeto (~200 hectares) precisa prever rendimento de safra e reagir a condições climáticas com agilidade. As aplicações vão do nível da propriedade (irrigação, escolha de cultura) até cadeias regionais e nacionais de abastecimento.

### 1.1.2. Descrição da Solução Desenvolvida

A solução analisa a base `crop_yield.csv` (cultura, precipitação, umidades, temperatura e rendimento) para: (i) explorar o comportamento dos dados; (ii) agrupar observações por tendência de produtividade com K-Means; (iii) treinar cinco algoritmos de regressão supervisionada para prever o rendimento em t/ha; e (iv) estimar o custo de hospedar a API do modelo na AWS, comparando São Paulo e Virgínia do Norte.

# <a name="c2"></a>2. Visão Geral do Projeto

## 2.1. Objetivos do Projeto

- Familiarizar-se com a base por meio de análise exploratória.
- Encontrar tendências de rendimento via clusterização e identificar outliers (IQR).
- Construir cinco modelos preditivos com algoritmos distintos e avaliá-los com MAE, MSE, RMSE e R².
- Comparar o custo On-Demand de uma instância Linux na AWS (2 vCPUs, 1 GiB, 5 Gbps, 50 GB) entre `sa-east-1` e `us-east-1`, justificando a região à luz de latência e restrições legais.

## 2.2. Público-Alvo

Gestores e agrônomos da fazenda cliente da FarmTech Solutions, além da equipe técnica que irá operar a API de inferência. Secundariamente, a banca da FIAP que avalia as entregas da Fase 5.

## 2.3. Metodologia

1. Carregamento e validação do dataset.
2. EDA (distribuições, correlação, boxplots e relações clima × rendimento).
3. Detecção de outliers pelo método IQR de Tukey.
4. Escalonamento, escolha de k (cotovelo + silhueta) e K-Means.
5. Split 80/20, one-hot da cultura, treino e avaliação dos cinco regressores.
6. Estimativa de custos AWS On-Demand e justificativa técnica da região.

# <a name="c3"></a>3. Desenvolvimento do Projeto

## 3.1. Tecnologias Utilizadas

Python 3.10+, pandas, NumPy, scikit-learn, matplotlib, seaborn, Jupyter Notebook, pytest e a calculadora/lista de preços da AWS (EC2 t3.micro + EBS gp3).

## 3.2. Modelagem e Algoritmos

- **Não supervisionado:** K-Means, com k escolhido por inércia e coeficiente de silhueta.
- **Supervisionado (regressão):** Linear Regression (baseline), Ridge (L2), Random Forest (bagging), Gradient Boosting (boosting) e SVR (kernel RBF). A diversidade de famílias de algoritmo atende ao enunciado e permite comparar viés/variância em relações climáticas não lineares.

## 3.3. Treinamento e Teste

O conjunto é dividido em 80% treino e 20% teste (`random_state=42`). Features numéricas são padronizadas (z-score) com o scaler ajustado apenas no treino. A cultura entra via one-hot encoding. Métricas: MAE, MSE, RMSE e R². Os módulos em `src/` são cobertos por 18 testes em `tests/`.

# <a name="c4"></a>4. Resultados e Avaliações

## 4.1. Análise dos Resultados

No dataset sintético incluído no repositório, o **Random Forest** obteve o melhor R² (0,945), seguido de SVR (0,941) e dos modelos lineares (0,939). Precipitação e temperatura aparecem como as variáveis mais importantes. A clusterização separa perfis climáticos distintos associados a diferentes níveis de rendimento. Cerca de 1–2% das observações foram marcadas como outliers de Yield.

Na nuvem, Virgínia do Norte é a opção mais barata (~US$ 11,59/mês contra ~US$ 19,86/mês em São Paulo). Ainda assim, a recomendação é **São Paulo (`sa-east-1`)**, por latência até os sensores no Brasil e por restrições legais (LGPD / dados que não podem sair do país).

## 4.2. Feedback dos Usuários

Não houve rodada formal de feedback com o cliente da fazenda nesta fase. A validação foi interna: execução ponta a ponta do notebook, suíte pytest e revisão dos critérios do enunciado (EDA, clusters, cinco algoritmos, README com custos AWS).

# <a name="c5"></a>5. Conclusões e Trabalhos Futuros

A solução atende às duas entregas obrigatórias da Fase 5: o pipeline de ML está funcional, testado e documentado no notebook; a comparação de custos AWS está no README, com justificativa técnica pela região de São Paulo. Pontos fortes: modularização, testes e clareza da análise. Pontos a melhorar: substituir o dataset sintético pela base oficial do portal; gravar os vídeos demonstrativos; e, se houver hardware, explorar o “Ir Além” com ESP32.

Plano futuro: validar o pipeline na base real, serializar o melhor modelo para a API, e revisar a cotação AWS na data da entrega oficial.

# <a name="c6"></a>6. Referências

- Enunciado oficial da Fase 5 — Capítulo 1 (disponível em `document/other/enunciado_original.txt`).
- Pedregosa et al. Scikit-learn: Machine Learning in Python. JMLR, 2011.
- AWS. Amazon EC2 On-Demand Pricing. https://aws.amazon.com/ec2/pricing/on-demand/
- Brasil. Lei nº 13.709/2018 (LGPD).

# <a name="c7"></a>Anexos

### Gráficos gerados pelo notebook

Os 11 gráficos da análise estão em `assets/graficos/` (distribuições, correlação, outliers, escolha de k, perfil dos clusters, comparação de modelos, previsão vs. real e importância das features).
