# Meu Projeto de Sistema de Recomendação de Filmes

## Objetivo

Bem-vindo ao meu Sistema de Recomendação de Filmes! Meu objetivo é fornecer a você recomendações de filmes personalizadas com base em várias características, como gênero, idioma original, popularidade e média de votos. Se você está procurando por novos filmes para assistir que estejam alinhados com seus gostos, estou aqui para ajudar.

## Roteiro

### 1. Pré-processamento de Dados

Antes de começar, realizei uma etapa crucial de pré-processamento de dados. Importei as bibliotecas necessárias e carreguei minha base de dados contendo informações sobre diversos filmes. Em seguida, fiz ajustes como renomear colunas, converter formatos de data e tratar valores inconsistentes para garantir que os dados estejam limpos e prontos para análise.

```python
# Importando bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import json

# Carregando base de dados
df = pd.read_csv('tmdb_5000_movies.csv')

# Renomeando colunas
traducao = {
    'budget': 'Orcamento',
    'genres': 'Gêneros',
    'original_language': 'Idioma original',
    'original_title': 'Título original',
    'popularity': 'Popularidade',
    'revenue': 'Receita',
    'runtime': 'Duração',
    'vote_average': 'Media_de_votos',
    'vote_count': 'Contagem_de_votos'
}
df = df.rename(columns=traducao)
```

### 2. Criação da Matriz de Características

Para construir um sistema de recomendação eficiente, precisei criar uma matriz de características que capture as informações relevantes sobre os filmes. Iniciei com a codificação one-hot para variáveis categóricas, transformando gêneros e idiomas em representações binárias. Além disso, apliquei o escalonamento Min-Max para normalizar as variáveis numéricas, assegurando que todas as características tenham o mesmo peso.

```python
# Codificação one-hot para variáveis categóricas
colunas_codi_one_hot = ['Gêneros', 'Idioma original']
for cols in colunas_codi_one_hot:
    coluna_one_hot = pd.get_dummies(df[cols].apply(pd.Series).stack(), dtype='str').groupby(level=0).sum()
    df = pd.concat([df, coluna_one_hot], axis=1)
    df = df.drop(cols, axis=1)

# Escalonamento Min-Max para variáveis numéricas
colunas_escala_min_max = ['Orcamento', 'Popularidade', 'Receita', 'Duração', 'Media_de_votos', 'Contagem_de_votos']
scaler = MinMaxScaler()
dados_escalonados = scaler.fit_transform(df[colunas_escala_min_max])
df[colunas_escala_min_max] = dados_escalonados
```

### 3. Cálculo da Similaridade entre Filmes

Aqui é onde a mágica acontece! Utilizei a similaridade do cosseno para medir o quão parecidos são os filmes com base nas características que criei. Quanto mais próximo o valor da similaridade do cosseno estiver de 1, mais semelhantes são os filmes. Essa etapa é fundamental para gerar recomendações precisas e relevantes.

```python
# Cálculo da similaridade do cosseno entre os filmes
similaridade_cosseno_filmes = cosine_similarity(df)
```

### 4. Implementação do Sistema de Recomendação

Agora é hora de colocar tudo em prática! Implementei um sistema de recomendação com base em modelos baseados em conteúdo usando o cálculo da similaridade do cosseno. Ao fornecer um filme de referência, meu sistema identifica os filmes mais similares e recomenda aqueles que têm maior compatibilidade com suas preferências.

```python
def recomendar_filmes(df, filme_referencia, num_recomendacoes):
    # Verificar se o filme de referência está presente no DataFrame
    if filme_referencia not in df.index:
        print(f"O filme '{filme_referencia}' não foi encontrado no DataFrame.")
        return None

    # Obter o índice numérico do filme de referência
    indice_filme_referencia = df.index.get_loc(filme_referencia)

    # Obter os índices dos filmes mais similares ao filme de referência
    filmes_similares = similaridade_cosseno_filmes[indice_filme_referencia].argsort()[:-num_recomendacoes-1:-1]

    # Filmes mais similares
    filmes_recomendados = df.index[filmes_similares].to_frame()
    filmes_recomendados.reset_index(drop=True, inplace=True)

    return filmes_recomendados

# Exemplo de uso
filme_referencia = 'The Dark Knight'
num_recomendacoes = 6
filmes_recomendados = recomendar_filmes(df, filme_referencia, num_recomendacoes)

filmes_recomendados[1:]
```

## Conclusão

Meu Sistema de Recomendação de Filmes é uma ferramenta poderosa para descobrir novos filmes alinhados com suas preferências. Ao combinar técnicas de processamento de dados e análise de similaridade, proporciono recomendações personalizadas que tornam sua experiência cinematográfica ainda mais empolgante e diversificada. Aproveite o mundo do entretenimento com minhas sugestões cuidadosamente selecionadas!




## Contribuição

Contribuições são bem-vindas! Se você deseja contribuir para este projeto, siga as diretrizes de colaboração e envie uma solicitação de pull.

## Contato

Se tiver alguma dúvida ou sugestão, sinta-se à vontade para entrar em contato:

- E-mail: miguelsilvafreire@hotmail.com
- Linkedin: (https://www.linkedin.com/in/miguel-freire99/)
