[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/ARkoM8Jo)
# Diagnóstico de retomada - Aprendizado de Máquina

Esta atividade serve para mapear o que você já domina em Aprendizado de Máquina depois das atividades anteriores da disciplina.

Responda individualmente. Use suas palavras. Rode o código quando possível. Se usar IA depois da primeira tentativa, registre o uso na seção 8.

Prazo: 11/05/2026 às 23:59, horário de Fortaleza.

## 1. Mapa do que eu lembro

Marque cada tópico como: lembro bem, lembro parcialmente, não lembro, nunca vi ou não tenho certeza.

- vetores, matrizes e produto escalar: lembro parcialmente
- média, desvio padrão e correlação: lembro bem
- probabilidade condicional e Teorema de Bayes: lembro parcialmente
- regressão linear: lembro bem
- classificação supervisionada: lembro parcialmente
- treino, teste e validação: lembro bem
- normalização ou padronização de dados: lembro parcialmente
- KNN: não lembro
- árvore de decisão: lembro bem
- matriz de confusão: lembro parcialmente
- acurácia, precisão, recall e F1-score: lembro parcialmente
- overfitting e underfitting: lembro parcialmente
- validação cruzada: lembro parcialmente
- Random Forest: lembro bem
- XGBoost ou boosting: lembro parcialmente
- `predict_proba()`: lembro parcialmente
- SQL/ETL aplicado a dados: lembro parcialmente
- simulação de Monte Carlo: lembro parcialmente

## 2. O que foi trabalhado antes

Explique, em 8 a 12 linhas:

1. quais desses tópicos você lembra de ter trabalhado na disciplina;
2. quais atividades ou exemplos você lembra;
3. o que você conseguiu fazer com autonomia;
4. o que você só conseguiu fazer seguindo roteiro;
5. qual assunto precisa ser retomado com mais urgência.

> Lembro de ter trabalhado com os seguintes tópicos: desvio padrão, probabilidade condicional, regressão linear, classificação supervisionada, treino, teste e validação, árvore de decisão, matriz de confusão, acurácia, precisão e F1, Random Forest e Simulação de Monte Carlo. Lembro da atividade para criar um modelo de regressão linear que prevê o preço do aluguel de um imóvel com base nas características do local, além de outra atividade que consistia em criar um modelo de predição de resultados de partidas de futebol com base em Random Forest ou XGBoost por meio de uma database SQLite, e utilizá-lo para prever o vencedor de um campeonato com base na simulação de Monte Carlo. Com autonomia foi possível construir a base do treinamento dos modelos (separar os dados em teste e treino, obter estatísticas do desempenho, etc). Houve mais dificuldade na hora de trabalhar com os bancos de dados relacionais e realizar tarefas mais estatísticas (como a simulação de Monte Carlo). Ao meu ver, o assunto que deve ser tratado com mais prioridade é a questão dos indicadores (matriz de confusão, acurácia, overfitting/underfitting, etc) e o conteúdo de classificação em si, pois foi explicado de maneira mais simplória e resumida.

## 3. Conceitos essenciais

Responda com suas palavras e dê um exemplo simples.

1. O que é aprendizado supervisionado?
> Nesse tipo de aprendizado, os dados que são fornecidos ao modelo vêm acompanhados de rótulos (podemos fornecer a "reposta correta" ao modelo).
2. O que é uma tarefa de classificação?
> Consiste no modelo classificar classes com base em seus atributos (Ex: Um modelo que diz se uma fruta é uma maçã ou uma laranja).
3. O que são features e target?
> Features são as características das classes a serem classificadas (Ex: peso, cor, formato) e o Target é o rótulo que será dado pelo modelo.
4. Para que serve separar treino e teste?
> Evitar o overfitting, que poderia causar um "viés" no modelo, prejudicando sua capacidade de predição.
5. O que é overfitting?
> Isso ocorre quando o modelo aprende "demais" os dados de treino, absorvendo os padrões e ruídos no conjunto de dados.
6. Por que acurácia pode ser uma métrica enganosa?
> A acurácia não é a melhor métrica para se medir o desempenho do modelo, principalmente quando se trata de um cojunto de dados desbalanceado. Isso ocorre pois como algumas classes aparecem mais do que outras, o modelo vai ter menos margem de erro.

## 4. Diagnóstico prático com Scikit-Learn

No arquivo `diagnostico_ml.py`, use o dataset `load_breast_cancer` do Scikit-Learn e faça:

1. carregue os dados;
2. separe `X` e `y`;
3. divida em treino e teste;
4. treine uma regressão logística;
5. treine uma árvore de decisão;
6. mostre matriz de confusão, acurácia, precisão, recall e F1-score para cada modelo;
7. compare o desempenho em treino e teste;
8. escreva aqui qual modelo generalizou melhor e por quê.

Se não conseguir terminar tudo, registre até onde chegou e qual erro apareceu.

### Resultados

Cole aqui os principais resultados do seu código.

```text
=== Regressão logística ===
Acurácia treino: 0.958
Acurácia teste: 0.958
Precisão teste: 0.947
Recall teste: 0.989
F1-score teste: 0.967
Matriz de confusão:
[[48  5]
 [ 1 89]]
Probabilidades das 5 primeiras amostras de teste:
[[0.01854698 0.98145302]
 [0.99816861 0.00183139]
 [0.17716357 0.82283643]
 [0.2344741  0.7655259 ]
 [0.19796601 0.80203399]]

=== Árvore de decisão ===
Acurácia treino: 1.000
Acurácia teste: 0.923
Precisão teste: 0.954
Recall teste: 0.922
F1-score teste: 0.938
Matriz de confusão:
[[49  4]
 [ 7 83]]
Probabilidades das 5 primeiras amostras de teste:
[[0. 1.]
 [1. 0.]
 [1. 0.]
 [0. 1.]
 [0. 1.]]  
```

### Interpretação

Qual modelo generalizou melhor? Explique usando as métricas e a comparação entre treino e teste.

Resposta:
> O modelo de Regressão Logística generalizou melhor. Ficou claro que o modelo de Árvore de Decisão sofreu overfitting, apresentando uma acurácia de 100%, mas geralmente desempenhando pior nos outros testes em comparação com a Regressão Logística (principalmente em F1).

## 5. Probabilidade e interpretação

Escolha um dos modelos treinados e responda:

1. O modelo produz probabilidade com `predict_proba()`?
2. O que significa uma probabilidade alta para uma classe?
3. Probabilidade alta garante que a previsão está correta? Explique.
4. Em um problema real, qual seria o risco de confiar cegamente nessa previsão?

Resposta:

> Tratando da Regressão Linear, o modelo produz probabilidade com `predict_proba()`. A probabilidade alta significa que aquela instância específica tem mais chance de pertencer àquela classe. Entretanto, probabilidade alta nem sempre garante que a previsão, apenas indica que o modelo tem uma alta taxa de confiança para aquela predição. Num problema real, confiar nessa previsão poderia trazer diversos prejuízos, de acordo com o escopo do problema, como prejudicar a integridade de uma pesquisa.

## 6. Generalização

Compare treino e teste:

1. Há sinal de overfitting?
2. Há sinal de underfitting?
3. O que você tentaria mudar para melhorar o resultado?
4. O que você precisaria estudar melhor para responder com mais segurança?

Resposta:
> Se tratando da Regressão Logística, não aparenta ter overfitting e nem underfitting. A fim de melhorar o resultado, talvez adicionar mais features e manipular hiperparâmetros. Para responder com mais segurança, precisaria entender melhor como funcionam os hiperparâmetros de cada modelo.

## 7. Ponto de dificuldade

Escolha um tópico da lista inicial e escreva:

1. o que você entende dele;
2. onde você se confunde;
3. que tipo de explicação ajudaria: exemplo no quadro, notebook guiado, exercício curto, revisão matemática, visualização ou projeto pequeno.

Resposta:
**Matriz de Confusão**
> A matriz de confusão serve como métrica para identificar os resultados corretos e equivocados do modelo. Mas me confundo em como eles os resultados são obtidos e como identificar as 'zonas' da matriz, principalmente quando é uma matriz maior que 2x2. Uma forma de explicação útil seria um exemplo no quadro ou visualização.

## 8. Uso de IA, se houver

Se você usou IA depois da primeira tentativa, registre:

```text
Pergunta feita:
Resumo da resposta:
Como eu verifiquei:
O que eu alterei na minha resposta:
O que ainda não entendi:
```
> Não utilizei IA, mas consultei o livro Mãos à Obra: Aprendizado de Máquina com Scikit-Learn & TensorFlow do autor Aurélien Géron

## Submissão no Moodle

Depois de finalizar, copie no Moodle:

```text
Repositório: https://github.com/frndchagas-org/diagn-stico-de-retomada-aprendizado-de-m-quina-augme06.git
Commit final: 
Autoavaliação: nível atual (Classificação), maior dificuldade (Matriz de Confusão) e tópico que precisa ser retomado (Métricas de Desempenho).
```
![Extra](https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExMWFwb250OTJiZjVpdHN5MTMyc20wbjlxaWw3c3V1OGI4cThjcXh4NCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/G6TgcESZt8FFk8XV7K/giphy.gif)