import pandas as pd
import numpy as np

# 1. Configuração dos parâmetros, faixas ideais e pesos (baseado na imagem)
parametros_regras = {
    'pH': {
        'faixa_ideal': (5.8, 6.2),
        'peso': 5,
        'faixa_geracao': (5.0, 7.0) # Faixa mais ampla para gerar valores bons e ruins
    },
    '% Sólidos': {
        'faixa_ideal': (30, 40),
        'peso': 2,
        'faixa_geracao': (25, 45)
    },
    'Dosagem (g/t)': {
        'faixa_ideal': (15, 30),
        'peso': 1,
        'faixa_geracao': (10, 35)
    }
}

# Parâmetros que são resultados, vamos apenas gerá-los aleatoriamente
parametros_resultado = {
    'Turbidez (FTU)': (20, 500),
    'Vs (m/h)': (0.5, 2.5)
}

# Quantidade de amostras a serem geradas
num_amostras = 500

# 2. Função para aplicar a lógica e classificar o resultado
def classificar_resultado(linha):
    """Calcula a pontuação e retorna a classificação baseada nas regras."""
    pontuacao = 0
    
    # Verifica cada parâmetro de entrada
    for param, config in parametros_regras.items():
        valor = linha[param]
        min_ideal, max_ideal = config['faixa_ideal']
        if not (min_ideal <= valor <= max_ideal):
            pontuacao += config['peso']
            
    # Aplica a lógica de classificação
    if pontuacao >= 5:
        return 'Falha'
    elif 1 <= pontuacao < 5:
        return 'Talvez'
    else: # pontuacao == 0
        return 'Sucesso'

# 3. Geração dos Dados
print(f"Gerando {num_amostras} amostras de dados sintéticos...")

# Dicionário para armazenar os dados gerados
dados = {}

# Gera os dados para os parâmetros de entrada
for param, config in parametros_regras.items():
    min_gen, max_gen = config['faixa_geracao']
    dados[param] = np.random.uniform(min_gen, max_gen, num_amostras)

# Gera os dados para os parâmetros de resultado
for param, (min_val, max_val) in parametros_resultado.items():
    dados[param] = np.random.uniform(min_val, max_val, num_amostras)
    
# Cria o DataFrame
df = pd.DataFrame(dados)

# Arredonda os valores para melhor visualização
df = df.round(2)

# 4. Aplicação das regras para criar a coluna "Resultado"
df['Resultado'] = df.apply(classificar_resultado, axis=1)

# Reordena as colunas para deixar o resultado no final
colunas_principais = list(parametros_regras.keys()) + list(parametros_resultado.keys())
df = df[colunas_principais + ['Resultado']]


# 5. Análise e Salvamento
print("\n5 primeiras linhas do dataset gerado:")
print(df.head())

print("\nDistribuição dos resultados:")
print(df['Resultado'].value_counts())

# Salva o arquivo em CSV
nome_arquivo = 'dataset_gerado_regras.csv'
df.to_csv(nome_arquivo, index=False)

print(f"\nBase de dados salva com sucesso no arquivo '{nome_arquivo}'!")