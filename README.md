# FIAP - Faculdade de Informática e Administração Paulista

<p align="center">
<a href= "https://www.fiap.com.br/"><img src="assets/logo-fiap.png" alt="FIAP - Faculdade de Informática e Admnistração Paulista" border="0" width=40% height=40%></a>
</p>

<br>

# FarmTech Solutions — Fase 5

## Graduação ON em Inteligência Artificial

## 👨‍🎓 Integrantes:
- <a href="https://www.linkedin.com/in/renan-ramos-571466a9/">Renan Ramos — RM573201</a>

## 👩‍🏫 Professores:
### Tutor(a)
- <a href="https://www.linkedin.com/in/sabrina-otoni-22525519b/">Sabrina Otoni</a>
### Coordenador(a)
- <a href="https://www.linkedin.com/in/andregodoichiovato/">André Godoi</a>


## 📜 Descrição

A FarmTech Solutions presta serviços de Inteligência Artificial para uma fazenda de médio porte (~200 hectares). Nesta fase o time trabalha com a base `crop_yield.csv` para explorar tendências de produtividade e prever o rendimento de safra.

Há duas entregas obrigatórias:

1. **Machine Learning** — análise exploratória, clusterização e cinco modelos de regressão supervisionada. O passo a passo, os achados e a discussão completa estão **somente no notebook Jupyter**. Este README não replica esse conteúdo.
2. **Cloud Computing** — estimativa de custos AWS para hospedar a API do modelo (documentada mais abaixo, neste README).

### Comece por aqui — Notebook Jupyter (Entrega 1)

O relatório completo da solução de Machine Learning (EDA, clusterização, outliers, cinco modelos e conclusões) está no Jupyter. Abra uma das opções abaixo.

| Versão | Arquivo no repositório | Abrir no Google Colab |
|---|---|---|
| Principal | [`src/RenanRamos_RM573201_pbl_fase5.ipynb`](src/RenanRamos_RM573201_pbl_fase5.ipynb) | [Abrir no Colab](https://colab.research.google.com/github/RenanSluder/templateFiapVfinal/blob/main/src/RenanRamos_RM573201_pbl_fase5.ipynb) |
| Web / Colab | [`src/RenanRamos_RM573201_pbl_fase5_web.ipynb`](src/RenanRamos_RM573201_pbl_fase5_web.ipynb) | [Abrir no Colab](https://colab.research.google.com/github/RenanSluder/templateFiapVfinal/blob/main/src/RenanRamos_RM573201_pbl_fase5_web.ipynb) |

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1r0LgJtEA8G5SFdSWksU7YYj4qQ3tyypI)

- Colab (GitHub — notebook principal): https://colab.research.google.com/github/RenanSluder/templateFiapVfinal/blob/main/src/RenanRamos_RM573201_pbl_fase5.ipynb
- Colab (GitHub — versão web): https://colab.research.google.com/github/RenanSluder/templateFiapVfinal/blob/main/src/RenanRamos_RM573201_pbl_fase5_web.ipynb
- Colab (Google Drive): https://colab.research.google.com/drive/1r0LgJtEA8G5SFdSWksU7YYj4qQ3tyypI

**Vídeo de demonstração (Entrega 1) — YouTube, não listado:** https://youtu.be/TkqFthurm0o



## 📁 Estrutura de pastas

Dentre os arquivos e pastas presentes na raiz do projeto, definem-se:

- <b>.github</b>: Nesta pasta ficarão os arquivos de configuração específicos do GitHub que ajudam a gerenciar e automatizar processos no repositório.

- <b>assets</b>: aqui estão os arquivos relacionados a elementos não-estruturados deste repositório, como imagens.

- <b>config</b>: Posicione aqui arquivos de configuração que são usados para definir parâmetros e ajustes do projeto.

- <b>document</b>: aqui estão todos os documentos do projeto que as atividades poderão pedir. Na subpasta "other", adicione documentos complementares e menos importantes.

- <b>scripts</b>: Posicione aqui scripts auxiliares para tarefas específicas do seu projeto. Exemplo: deploy, migrações de banco de dados, backups.

- <b>src</b>: Todo o código fonte criado para o desenvolvimento do projeto ao longo das 7 fases.

- <b>README.md</b>: arquivo que serve como guia e explicação geral sobre o projeto (o mesmo que você está lendo agora).

## 🔧 Como executar o código

### Pré-requisitos

- Python 3.10+
- pip

### Instalação e abertura do notebook

```bash
git clone https://github.com/RenanSluder/templateFiapVfinal.git
cd templateFiapVfinal
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook src/RenanRamos_RM573201_pbl_fase5.ipynb
```

No Google Colab, use um dos links da seção acima (badge **Open In Colab**, GitHub ou Drive). Alternativa: `Arquivo → Fazer upload de notebook` com `src/RenanRamos_RM573201_pbl_fase5_web.ipynb`, depois `Ambiente de execução → Executar tudo`.

Testes dos módulos de suporte (opcional): `pytest tests/ -v`

A partir daqui, todo o passo a passo da solução de Machine Learning está no Jupyter.

---

## ☁️ Entrega 2 — Cloud Computing

### Cenário

A API que recebe os dados dos sensores (variáveis climáticas usadas no notebook) e roda o modelo de Machine Learning precisa ser hospedada em uma máquina Linux simples na AWS, com a configuração mínima:

- 2 CPUs
- 1 GiB de memória
- Até 5 Gigabit de rede
- 50 GB de armazenamento (HD)

Essa configuração corresponde à instância **`t3.micro`** (2 vCPUs, 1 GiB RAM, rede até 5 Gbps), com um volume **EBS gp3 de 50 GB**.

### 1. Comparação de custos: São Paulo (BR) vs. Virgínia do Norte (EUA)

Preços **On-Demand (100%)**, Linux, obtidos na lista pública da AWS (consulta em set/2026):

| Item | US East (N. Virginia) `us-east-1` | South America (São Paulo) `sa-east-1` |
|---|---:|---:|
| EC2 `t3.micro` (USD/hora) | $0.0104 | $0.0168 |
| EC2 `t3.micro` (USD/mês, 730h) | $7.59 | $12.26 |
| EBS `gp3` 50 GB (USD/GB-mês) | $0.0800 | $0.1520 |
| EBS `gp3` 50 GB (USD/mês) | $4.00 | $7.60 |
| **Total mensal estimado** | **$11.59** | **$19.86** |
| **Total anual estimado** | **$139.10** | **$238.37** |

**Diferença:** São Paulo é aproximadamente **71% mais cara** que Virgínia do Norte — **$8.27/mês** (**$99.24/ano**).

> 💡 Valores com 730 horas/mês (instância ligada continuamente), padrão da calculadora AWS On-Demand.

#### Gráfico comparativo de custos

```
Custo mensal estimado (USD) — instância t3.micro + 50 GB EBS gp3

N. Virginia (us-east-1)  ████████████░░░░░░░░  $11.59
São Paulo   (sa-east-1)  ████████████████████  $19.86
                          0        10        20
```

### 2. Qual opção escolher? Acesso rápido aos dados + restrições legais

- **Restrição legal de armazenar dados no exterior:** se a fazenda opera no Brasil e há exigência regulatória (LGPD ou política interna) de que os dados não saiam do país, `sa-east-1` deixa de ser “mais cara” e passa a ser a **única opção viável**.
- **Latência de acesso aos dados dos sensores:** os sensores estão na fazenda no Brasil. Uma instância em `sa-east-1` está geograficamente muito mais próxima do que `us-east-1`, reduzindo a latência de cada requisição da API.

**Conclusão e justificativa técnica:** apesar do custo ~71% maior, a opção recomendada é hospedar a API em **São Paulo (`sa-east-1`)**. A diferença (~$8/mês) é pequena em termos absolutos e é superada por dois fatores: (1) a restrição legal de manter os dados no país e (2) a menor latência para operação em tempo real. Economizar ~$100/ano não compensa o risco de não conformidade nem a degradação de performance.

### Vídeo de demonstração (Entrega 2)

**[Link a preencher após a gravação — YouTube, não listado, até 5 minutos, demonstrando a comparação de custos na calculadora AWS]**

---

## 🚀 Ir Além (opcional, não vale nota no boletim)

Este capítulo não implementa as opções de "Ir Além" (ESP32 real + Wi-Fi, ou classificação de saúde de plantação em tempo real), por não depender de hardware físico disponível para este projeto.


## 🗃 Histórico de lançamentos

* 0.5.0 - 08/09/2026
    * Adequação da Fase 5 (Cap. 1) à estrutura do template FIAP da raiz: notebooks e módulos em `src/`, documentação em `document/`, configurações em `config/`, scripts auxiliares e README no formato oficial.
* 0.1.0 - 08/09/2026
    * Entrega inicial: EDA, K-Means, cinco modelos de regressão, testes automatizados e estimativa de custos AWS.

## 📋 Licença

<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/agodoi/template">MODELO GIT FIAP</a> por <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://fiap.com.br">Fiap</a> está licenciado sobre <a href="http://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Attribution 4.0 International</a>.</p>
