# src/data_loader.py
import pandas as pd

def load_data(path):
    """Carrega os dados de um arquivo CSV."""
    try:
        df = pd.read_csv(path)
        print(f"Dados carregados com sucesso de {path}")
        return df
    except FileNotFoundError:
        print(f"Erro: O arquivo {path} não foi encontrado.")
        return None