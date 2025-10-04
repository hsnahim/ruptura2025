# config.py
# Arquivo de configuração ATUALIZADO para o dataset gerado por regras

# --- Caminhos de Arquivos ---
# ATUALIZE O NOME DO ARQUIVO AQUI
DATA_PATH = "data/dataset_gerado_regras.csv" 
MODEL_SAVE_PATH = "models/"

# --- Configurações das Colunas 🎯 ---
# ATUALIZE AS COLUNAS ALVO AQUI
# Coluna alvo para o problema de REGRESSÃO
# (Continua sendo uma boa escolha)
TARGET_REGRESSION = 'Vs (m/h)'

# Coluna alvo para o problema de CLASSIFICAÇÃO
# (Esta é a principal mudança)
TARGET_CLASSIFICATION = 'Resultado'

# --- Configurações dos Modelos ---
TREE_MAX_DEPTH_1 = 3
TREE_MAX_DEPTH_2 = 4
RANDOM_STATE = 42 # Para garantir que os resultados sejam reprodutíveis